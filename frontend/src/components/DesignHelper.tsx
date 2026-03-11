"use client";
import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import * as api from "@/api";
import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface Props {
  datasetType?: string;
  currentArchitecture?: api.Architecture | null;
  messages: ChatMessage[];
  onMessagesChange: (messages: ChatMessage[]) => void;
}

export default function DesignHelper({
  datasetType,
  currentArchitecture,
  messages,
  onMessagesChange,
}: Props) {
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  async function handleChatSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;

    const userMessage = chatInput.trim();
    setChatInput("");
    onMessagesChange([...messages, { role: "user", content: userMessage }]);
    setChatLoading(true);

    try {
      const result = await api.getDesignHelp({
        message: userMessage,
        context: {
          dataset_type: datasetType || "image",
          current_architecture: currentArchitecture,
        },
      });
      onMessagesChange([
        ...messages,
        { role: "user", content: userMessage },
        { role: "assistant", content: result.response },
      ]);
    } catch {
      onMessagesChange([
        ...messages,
        { role: "user", content: userMessage },
        {
          role: "assistant",
          content: "Sorry, I encountered an error. Please try again.",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  return (
    <Paper
      variant="outlined"
      sx={{
        borderColor: "divider",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <Box
        sx={{
          maxHeight: 300,
          overflowY: "auto",
          p: 1.5,
          flex: 1,
        }}
      >
        {messages.length === 0 && (
          <Box sx={{ textAlign: "center", py: 2, px: 1 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Ask me anything about neural network design:
            </Typography>
            <Typography
              component="ul"
              variant="body2"
              color="text.secondary"
              sx={{ pl: 2.5, m: 0, textAlign: "left", "& li": { mb: 0.5 } }}
            >
              <li>Layer types and when to use them</li>
              <li>Training parameters</li>
              <li>Architecture patterns</li>
              <li>Troubleshooting tips</li>
            </Typography>
          </Box>
        )}
        {messages.map((msg, i) => (
          <Box
            key={i}
            sx={{
              mb: 1.5,
              textAlign: msg.role === "user" ? "right" : "left",
            }}
          >
            <Box
              sx={{
                display: "inline-block",
                maxWidth: "85%",
                p: 1.25,
                borderRadius: 1,
                fontSize: "0.9375rem",
                lineHeight: 1.6,
                textAlign: "left",
                ...(msg.role === "user"
                  ? { bgcolor: "primary.main", color: "primary.contrastText" }
                  : {
                      bgcolor: "background.default",
                      border: "1px solid",
                      borderColor: "divider",
                    }),
              }}
            >
              {msg.role === "assistant" ? (
                <Markdown remarkPlugins={[remarkGfm]}>{msg.content}</Markdown>
              ) : (
                msg.content
              )}
            </Box>
          </Box>
        ))}
        {chatLoading && (
          <Box sx={{ mb: 1.5, textAlign: "left" }}>
            <Box
              sx={{
                display: "inline-block",
                p: 1.25,
                borderRadius: 1,
                border: "1px solid",
                borderColor: "divider",
                fontStyle: "italic",
                color: "text.secondary",
              }}
            >
              Thinking...
            </Box>
          </Box>
        )}
      </Box>
      <Box
        component="form"
        onSubmit={handleChatSubmit}
        sx={{
          display: "flex",
          gap: 1,
          p: 1,
          borderTop: "1px solid",
          borderColor: "divider",
          bgcolor: "background.default",
        }}
      >
        <TextField
          size="small"
          fullWidth
          placeholder="Ask about layers, training, etc..."
          value={chatInput}
          onChange={(e) => setChatInput(e.target.value)}
          disabled={chatLoading}
          sx={{
            "& .MuiOutlinedInput-root": {
              bgcolor: "action.hover",
            },
          }}
        />
        <Button
          type="submit"
          variant="contained"
          disabled={!chatInput.trim() || chatLoading}
        >
          Send
        </Button>
      </Box>
    </Paper>
  );
}
