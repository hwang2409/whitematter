"use client";
import { Component, ReactNode } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    console.error("ErrorBoundary caught an error:", error);
    console.error("Component stack:", errorInfo.componentStack);
  }

  handleRetry = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Box
          sx={{
            minHeight: "60vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            p: 3,
          }}
        >
          <Box
            sx={{
              p: 4,
              maxWidth: 440,
              textAlign: "center",
              bgcolor: "#18181B",
              border: "1px solid #27272A",
              borderRadius: "12px",
              boxShadow: "0 16px 48px rgba(0,0,0,0.4)",
            }}
          >
            <Typography
              sx={{
                fontFamily: "'Instrument Serif', Georgia, serif",
                fontSize: "1.75rem",
                fontWeight: 400,
                color: "#FAFAFA",
                mb: 1,
                lineHeight: 1.2,
              }}
            >
              Something went wrong
            </Typography>
            <Typography
              sx={{
                fontSize: "0.875rem",
                fontFamily: "'Outfit', sans-serif",
                color: "#A1A1AA",
                mb: 2.5,
                lineHeight: 1.5,
              }}
            >
              An unexpected error occurred. Please try again or refresh the page.
            </Typography>
            {this.state.error && (
              <details
                style={{
                  marginBottom: 20,
                  textAlign: "left",
                  cursor: "pointer",
                  color: "#52525B",
                  fontSize: "0.8125rem",
                  fontFamily: "'Outfit', sans-serif",
                }}
              >
                <summary style={{ marginBottom: 8 }}>Error details</summary>
                <pre
                  style={{
                    padding: 12,
                    background: "#09090B",
                    border: "1px solid #27272A",
                    borderRadius: 8,
                    overflow: "auto",
                    fontSize: "0.75rem",
                    fontFamily: "'JetBrains Mono', monospace",
                    color: "#A1A1AA",
                    lineHeight: 1.5,
                    margin: 0,
                  }}
                >
                  {this.state.error.message}
                </pre>
              </details>
            )}
            <Box sx={{ display: "flex", gap: 1, justifyContent: "center", flexWrap: "wrap" }}>
              <Button
                variant="contained"
                onClick={this.handleRetry}
                sx={{
                  bgcolor: "#F97316",
                  color: "#fff",
                  fontFamily: "'Outfit', sans-serif",
                  fontSize: "0.8125rem",
                  fontWeight: 500,
                  textTransform: "none",
                  borderRadius: "8px",
                  boxShadow: "none",
                  "&:hover": { bgcolor: "#EA580C", boxShadow: "none" },
                }}
              >
                Try Again
              </Button>
              <Button
                variant="outlined"
                onClick={() => window.location.reload()}
                sx={{
                  borderColor: "#27272A",
                  color: "#A1A1AA",
                  fontFamily: "'Outfit', sans-serif",
                  fontSize: "0.8125rem",
                  fontWeight: 500,
                  textTransform: "none",
                  borderRadius: "8px",
                  "&:hover": {
                    borderColor: "#3F3F46",
                    color: "#FAFAFA",
                    bgcolor: "rgba(255,255,255,0.03)",
                  },
                }}
              >
                Refresh Page
              </Button>
            </Box>
          </Box>
        </Box>
      );
    }

    return this.props.children;
  }
}
