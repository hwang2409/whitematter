"use client";
import NextLink from "next/link";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import AutoAwesomeOutlined from "@mui/icons-material/AutoAwesomeOutlined";
import ShowChartOutlined from "@mui/icons-material/ShowChartOutlined";
import RocketLaunchOutlined from "@mui/icons-material/RocketLaunchOutlined";
import GoogleSignInButton from "@/components/GoogleSignInButton";

const FEATURES = [
  {
    icon: <AutoAwesomeOutlined sx={{ fontSize: 32, color: "primary.main" }} />,
    title: "AI Architecture Designer",
    description:
      "Describe what you want to build. Claude designs the neural network.",
  },
  {
    icon: <ShowChartOutlined sx={{ fontSize: 32, color: "primary.main" }} />,
    title: "Live Training",
    description:
      "Watch your model train in real-time with live loss and accuracy charts.",
  },
  {
    icon: <RocketLaunchOutlined sx={{ fontSize: 32, color: "primary.main" }} />,
    title: "One-Click Deploy",
    description: "Deploy trained models as API endpoints instantly.",
  },
];

export default function LandingPage() {
  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "background.default" }}>
      <Container
        maxWidth="md"
        sx={{ pt: { xs: 8, md: 14 }, pb: 8, textAlign: "center" }}
      >
        <Typography
          component="span"
          sx={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: "1.5rem",
            fontWeight: 700,
            color: "primary.main",
            letterSpacing: "0.02em",
            display: "block",
            mb: 3,
          }}
        >
          wm
        </Typography>

        <Typography
          variant="h1"
          sx={{
            fontSize: { xs: "2rem", md: "3rem" },
            fontWeight: 700,
            letterSpacing: "-0.03em",
            mb: 2,
            color: "text.primary",
          }}
        >
          Train neural networks from your browser.
        </Typography>

        <Typography
          variant="h2"
          sx={{
            fontSize: { xs: "1.1rem", md: "1.35rem" },
            fontWeight: 400,
            color: "text.secondary",
            mb: 4,
            maxWidth: 560,
            mx: "auto",
          }}
        >
          Design architectures with AI. Deploy with one click.
        </Typography>

        <Box
          sx={{
            display: "flex",
            gap: 2,
            justifyContent: "center",
            flexWrap: "wrap",
            mb: 3,
          }}
        >
          <Box sx={{ display: "flex", gap: 2, flexDirection: { xs: "column", sm: "row" }, alignItems: "center" }}>
            <Button
              variant="contained"
              size="large"
              component={NextLink}
              href="/register"
              sx={{ px: 4, py: 1.5, fontSize: "1rem" }}
            >
              Get Started
            </Button>
            <GoogleSignInButton label="Sign in with Google" fullWidth={false} />
          </Box>
          <Button
            variant="outlined"
            size="large"
            component={NextLink}
            href="/login"
            sx={{ px: 4, py: 1.5, fontSize: "1rem" }}
          >
            Sign in
          </Button>
        </Box>

        <Box
          sx={{
            width: "100%",
            maxWidth: 720,
            mx: "auto",
            aspectRatio: "16/9",
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 2,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            bgcolor: "background.paper",
          }}
        >
          <Typography color="text.secondary" variant="body2">
            Demo GIF
          </Typography>
        </Box>
      </Container>

      <Container maxWidth="md" sx={{ pb: 10 }}>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", md: "repeat(3, 1fr)" },
            gap: 4,
          }}
        >
          {FEATURES.map((feature) => (
            <Box key={feature.title} sx={{ textAlign: "center" }}>
              {feature.icon}
              <Typography variant="h3" sx={{ mt: 1.5, mb: 1 }}>
                {feature.title}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {feature.description}
              </Typography>
            </Box>
          ))}
        </Box>
      </Container>

      <Box
        component="footer"
        sx={{
          py: 3,
          textAlign: "center",
          borderTop: "1px solid",
          borderColor: "divider",
        }}
      >
        <Typography variant="body2" color="text.secondary">
          Built with whitematter
        </Typography>
      </Box>
    </Box>
  );
}
