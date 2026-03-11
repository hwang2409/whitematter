"use client";
import Link from "next/link";
import { useAuth } from "@/context/AuthContext";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import CardContent from "@mui/material/CardContent";
import Stack from "@mui/material/Stack";
import DatasetOutlined from "@mui/icons-material/DatasetOutlined";
import SchoolOutlined from "@mui/icons-material/SchoolOutlined";
import PsychologyOutlined from "@mui/icons-material/PsychologyOutlined";
import SettingsOutlined from "@mui/icons-material/SettingsOutlined";

const links = [
  { href: "/data", label: "Data", sublabel: "Upload and manage datasets", icon: DatasetOutlined },
  { href: "/train", label: "Train", sublabel: "Train your model", icon: SchoolOutlined },
  { href: "/models", label: "Models", sublabel: "View and deploy models", icon: PsychologyOutlined },
  { href: "/settings", label: "Settings", sublabel: "AWS and preferences", icon: SettingsOutlined },
];

export default function DashboardPage() {
  const { user } = useAuth();

  return (
    <Box>
      <Typography variant="h2" sx={{ mb: 0.5 }}>
        Dashboard
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Welcome back, {user?.email}
      </Typography>
      <Stack direction={{ xs: "column", sm: "row" }} spacing={2} useFlexGap flexWrap="wrap">
        {links.map(({ href, label, sublabel, icon: Icon }) => (
          <Card key={href} variant="outlined" sx={{ flex: { sm: "1 1 200px" }, maxWidth: 320 }}>
            <CardActionArea component={Link} href={href} sx={{ height: "100%" }}>
              <CardContent sx={{ py: 2.5, "&:last-child": { pb: 2.5 } }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1 }}>
                  <Icon sx={{ color: "text.secondary", fontSize: 22 }} />
                  <Typography variant="subtitle1" fontWeight={600}>
                    {label}
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {sublabel}
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        ))}
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 3 }}>
        BYOC training jobs and model architectures can be listed here once the API is wired.
      </Typography>
    </Box>
  );
}
