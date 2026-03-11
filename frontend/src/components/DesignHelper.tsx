"use client";
import { useState } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import * as api from '@/api';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface Props {
  datasetType?: string;
  currentArchitecture?: api.Architecture | null;
  messages: ChatMessage[];
  onMessagesChange: (messages: ChatMessage[]) => void;
}

export default function DesignHelper({ datasetType, currentArchitecture, messages, onMessagesChange }: Props) {
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  async function handleChatSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;

    const userMessage = chatInput.trim();
    setChatInput('');
    onMessagesChange([...messages, { role: 'user', content: userMessage }]);
    setChatLoading(true);

    try {
      const result = await api.getDesignHelp({
        message: userMessage,
        context: {
          dataset_type: datasetType || 'image',
          current_architecture: currentArchitecture,
        },
      });
      onMessagesChange([...messages, { role: 'user', content: userMessage }, { role: 'assistant', content: result.response }]);
    } catch (e: any) {
      onMessagesChange([
        ...messages,
        { role: 'user', content: userMessage },
        { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' },
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  return (
    <>
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-welcome">
            <p>Ask me anything about neural network design:</p>
            <ul>
              <li>Layer types and when to use them</li>
              <li>Training parameters</li>
              <li>Architecture patterns</li>
              <li>Troubleshooting tips</li>
            </ul>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            <div className="message-content">
              {msg.role === 'assistant' ? (
                <Markdown remarkPlugins={[remarkGfm]}>{msg.content}</Markdown>
              ) : (
                msg.content
              )}
            </div>
          </div>
        ))}
        {chatLoading && (
          <div className="chat-message assistant">
            <div className="message-content typing">Thinking...</div>
          </div>
        )}
      </div>
      <form className="chat-input-form" onSubmit={handleChatSubmit}>
        <input
          type="text"
          value={chatInput}
          onChange={(e) => setChatInput(e.target.value)}
          placeholder="Ask about layers, training, etc..."
          disabled={chatLoading}
        />
        <button type="submit" className="btn primary" disabled={!chatInput.trim() || chatLoading}>
          Send
        </button>
      </form>
    </>
  );
}
