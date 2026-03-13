"use client";

import { useState } from "react";
import Box from "@mui/material/Box";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Typography from "@mui/material/Typography";
import GeneralTab from "@/components/settings/GeneralTab";
import AccountTab from "@/components/settings/AccountTab";
import BillingTab from "@/components/settings/BillingTab";
import ConnectTab from "@/components/settings/ConnectTab";

const TABS = ["General", "Account", "Billing", "Connect"] as const;

export default function SettingsRoute() {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <Box sx={{ p: { xs: 2, sm: 3 }, maxWidth: 900, mx: "auto", width: "100%" }}>
      <Typography variant="h2" sx={{ mb: 3 }}>
        Settings
      </Typography>

      <Box
        sx={{
          display: "flex",
          flexDirection: { xs: "column", sm: "row" },
          gap: 0,
          minHeight: 400,
        }}
      >
        {/* Vertical tabs */}
        <Tabs
          orientation="vertical"
          value={activeTab}
          onChange={(_, v) => setActiveTab(v)}
          sx={{
            minWidth: 180,
            borderRight: 1,
            borderColor: "divider",
            "& .MuiTab-root": {
              alignItems: "flex-start",
              textTransform: "none",
              fontSize: "0.875rem",
              minHeight: 40,
              py: 1,
            },
          }}
        >
          {TABS.map((label) => (
            <Tab key={label} label={label} />
          ))}
        </Tabs>

        {/* Tab content */}
        <Box sx={{ flex: 1, pl: { xs: 0, sm: 4 }, pt: { xs: 2, sm: 0 } }}>
          {activeTab === 0 && <GeneralTab />}
          {activeTab === 1 && <AccountTab />}
          {activeTab === 2 && <BillingTab />}
          {activeTab === 3 && <ConnectTab />}
        </Box>
      </Box>
    </Box>
  );
}
