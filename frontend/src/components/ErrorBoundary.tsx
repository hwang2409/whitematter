"use client";
import { Component, ReactNode } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Paper from "@mui/material/Paper";

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
          <Paper
            variant="outlined"
            sx={{
              p: 3,
              maxWidth: 440,
              textAlign: "center",
              borderColor: "divider",
            }}
          >
            <Typography variant="h2" sx={{ mb: 1 }}>
              Something went wrong
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              An unexpected error occurred. Please try again or refresh the page.
            </Typography>
            {this.state.error && (
              <details
                style={{
                  marginBottom: 16,
                  textAlign: "left",
                  cursor: "pointer",
                  color: "rgba(255,255,255,0.6)",
                  fontSize: "0.8125rem",
                }}
              >
                <summary>Error details</summary>
                <pre
                  style={{
                    marginTop: 8,
                    padding: 12,
                    background: "rgba(0,0,0,0.3)",
                    borderRadius: 8,
                    overflow: "auto",
                    fontSize: "0.75rem",
                    fontFamily: "'DM Mono', monospace",
                  }}
                >
                  {this.state.error.message}
                </pre>
              </details>
            )}
            <Box sx={{ display: "flex", gap: 1, justifyContent: "center", flexWrap: "wrap" }}>
              <Button variant="contained" color="primary" onClick={this.handleRetry}>
                Try Again
              </Button>
              <Button variant="outlined" onClick={() => window.location.reload()}>
                Refresh Page
              </Button>
            </Box>
          </Paper>
        </Box>
      );
    }

    return this.props.children;
  }
}
