import { fetchWithTimeout, API_BASE } from './client';
import type { Conversation, ChatMessage, TrainingStatus } from './types';
import { getStoredToken } from '@/services/auth';

function mapConversation(raw: any): Conversation {
  return {
    id: raw.id,
    title: raw.title,
    phase: raw.phase,
    isStarred: raw.is_starred ?? raw.isStarred ?? false,
    createdAt: raw.created_at ?? raw.createdAt ?? new Date().toISOString(),
    updatedAt: raw.updated_at ?? raw.updatedAt ?? raw.created_at ?? raw.createdAt ?? new Date().toISOString(),
  };
}

function mapMessage(raw: any): ChatMessage {
  return {
    id: raw.id ?? crypto.randomUUID(),
    role: raw.role,
    content: raw.content,
    type: raw.type ?? raw.message_type ?? "text",
    metadata: raw.metadata,
    createdAt: raw.created_at ?? raw.createdAt ?? new Date().toISOString(),
  };
}

export async function createConversation(token: string): Promise<Conversation> {
  const res = await fetchWithTimeout(`${API_BASE}/chat/conversations`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  });
  if (!res.ok) throw new Error("Failed to create conversation");
  const raw = await res.json();
  return mapConversation(raw);
}

export async function getConversations(): Promise<Conversation[]> {
  const token = getStoredToken();
  const res = await fetchWithTimeout(`${API_BASE}/chat/conversations`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  const data = await res.json();
  return (data.conversations ?? []).map(mapConversation);
}

export async function getConversation(
  token: string,
  id: string,
): Promise<{ conversation: Conversation; messages: ChatMessage[] }> {
  const res = await fetchWithTimeout(`${API_BASE}/chat/conversations/${id}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Failed to load conversation");
  const data = await res.json();
  return {
    conversation: mapConversation(data.conversation),
    messages: (data.messages ?? []).map(mapMessage),
  };
}

export async function updateConversation(
  id: string,
  data: { title?: string; isStarred?: boolean },
): Promise<Conversation> {
  const token = getStoredToken();
  const body: Record<string, unknown> = {};
  if (data.title !== undefined) body.title = data.title;
  if (data.isStarred !== undefined) body.is_starred = data.isStarred;
  const res = await fetchWithTimeout(`${API_BASE}/chat/conversations/${id}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
  });
  return res.json();
}

export async function deleteConversation(id: string): Promise<void> {
  const token = getStoredToken();
  await fetchWithTimeout(`${API_BASE}/chat/conversations/${id}`, {
    method: "DELETE",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
}

export function sendChatMessage(
  conversationId: string,
  content: string,
  onChunk: (text: string) => void,
  onDone: (message: ChatMessage) => void,
  onError: (error: Error) => void,
  token?: string,
  onToolUse?: (name: string) => void,
  onToolResult?: (name: string, summary: string) => void,
  onToolProgress?: (name: string, message: string, elapsed: number) => void,
): { abort: () => void } {
  const controller = new AbortController();
  token = token ?? getStoredToken() ?? undefined;

  (async () => {
    try {
      const res = await fetch(
        `${API_BASE}/chat/conversations/${conversationId}/messages`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify({ content }),
          signal: controller.signal,
        },
      );

      if (!res.ok) {
        throw new Error(`Chat error: ${res.status}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") {
              onDone({
                id: crypto.randomUUID(),
                role: "assistant",
                content: fullText,
                type: "text",
                createdAt: new Date().toISOString(),
              });
              return;
            }
            try {
              const parsed = JSON.parse(data);
              if (parsed.type === "chunk") {
                fullText += parsed.content;
                onChunk(parsed.content);
              } else if (parsed.type === "tool_use") {
                onToolUse?.(parsed.name);
              } else if (parsed.type === "tool_progress") {
                onToolProgress?.(parsed.name, parsed.message, parsed.elapsed);
              } else if (parsed.type === "tool_result") {
                onToolResult?.(parsed.name, parsed.summary);
              } else if (parsed.type === "done") {
                const msg = mapMessage(parsed.message);
                if (parsed.phase) {
                  msg.metadata = { ...msg.metadata, phase: parsed.phase };
                }
                onDone(msg);
                return;
              }
            } catch {
              /* ignore parse errors for partial lines */
            }
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        onError(err as Error);
      }
    }
  })();

  return { abort: () => controller.abort() };
}

export async function getChatTrainingStatus(conversationId: string): Promise<TrainingStatus> {
  const token = getStoredToken();
  const res = await fetchWithTimeout(`${API_BASE}/chat/conversations/${conversationId}/training`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  return res.json();
}
