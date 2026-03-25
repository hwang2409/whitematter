"use client";

import { useState } from "react";
import Box from "@mui/material/Box";
import AccountTab from "@/components/settings/AccountTab";
import BillingTab from "@/components/settings/BillingTab";
import ConnectTab from "@/components/settings/ConnectTab";

const TABS = ["Account", "Billing", "Connect"] as const;

export default function SettingsRoute() {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <Box
      sx={{
        p: { xs: 2, sm: 4, md: 5 },
        maxWidth: 820,
        mx: "auto",
        width: "100%",
      }}
    >
      {/* Tab bar */}
      <Box
        sx={{
          display: "flex",
          gap: 0,
          mb: 5,
          borderBottom: "1px solid #27272A",
        }}
      >
        {TABS.map((label, i) => (
          <Box
            key={label}
            onClick={() => setActiveTab(i)}
            sx={{
              px: 2,
              py: 1.25,
              fontSize: "0.8125rem",
              fontFamily: "'Outfit', sans-serif",
              fontWeight: activeTab === i ? 500 : 400,
              color: activeTab === i ? "#FAFAFA" : "#A1A1AA",
              cursor: "pointer",
              borderBottom: "2px solid",
              borderColor: activeTab === i ? "#F97316" : "transparent",
              mb: "-1px",
              transition: "color 0.2s ease, border-color 0.2s ease",
              userSelect: "none",
              "&:hover": {
                color: "#FAFAFA",
              },
            }}
          >
            {label}
          </Box>
        ))}
      </Box>

      {/* Content */}
      <Box
        sx={{
          animation: "settingsFadeIn 0.2s ease-out",
          "@keyframes settingsFadeIn": {
            from: { opacity: 0, transform: "translateY(4px)" },
            to: { opacity: 1, transform: "translateY(0)" },
          },
        }}
        key={activeTab}
      >
        {activeTab === 0 && <AccountTab />}
        {activeTab === 1 && <BillingTab />}
        {activeTab === 2 && <ConnectTab />}
      </Box>
    </Box>
  );
}
