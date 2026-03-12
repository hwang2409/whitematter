"use client";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import HelpOutlineOutlined from "@mui/icons-material/HelpOutlineOutlined";
import { PARAM_TOOLTIPS } from "@/lib/paramTooltips";

interface Props {
  paramKey: string;
}

export default function ParamTooltip({ paramKey }: Props) {
  const data = PARAM_TOOLTIPS[paramKey];
  if (!data) return null;

  return (
    <Tooltip
      title={
        <Box sx={{ p: 0.5 }}>
          <Typography variant="body2" sx={{ mb: 0.5 }}>
            {data.description}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Recommended: {data.range}
          </Typography>
        </Box>
      }
      arrow
      placement="top"
    >
      <IconButton
        size="small"
        aria-label={`Info about ${paramKey.replace(/_/g, " ")}`}
        sx={{ ml: 0.5, p: 0.25, color: "text.secondary" }}
      >
        <HelpOutlineOutlined sx={{ fontSize: 16 }} />
      </IconButton>
    </Tooltip>
  );
}
