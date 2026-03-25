"use client";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import TextField from "@mui/material/TextField";
import InputAdornment from "@mui/material/InputAdornment";
import SearchOutlined from "@mui/icons-material/SearchOutlined";
import RefreshOutlined from "@mui/icons-material/RefreshOutlined";

interface ModelFiltersProps {
  modelSearch: string;
  onSearchChange: (value: string) => void;
  onRefresh: () => void;
}

export default function ModelFilters({ modelSearch, onSearchChange, onRefresh }: ModelFiltersProps) {
  return (
    <>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2.5 }}>
        <Typography
          variant="h2"
          sx={{
            fontFamily: "'Instrument Serif', Georgia, serif",
            fontSize: "1.75rem",
            fontWeight: 400,
            color: "#FAFAFA",
            letterSpacing: "-0.01em",
          }}
        >
          Your Models
        </Typography>
        <IconButton
          onClick={onRefresh}
          size="small"
          sx={{
            color: "#A1A1AA",
            border: "1px solid #27272A",
            borderRadius: "8px",
            width: 36,
            height: 36,
            transition: "all 0.15s ease",
            "&:hover": {
              color: "#FAFAFA",
              borderColor: "#3F3F46",
              bgcolor: "rgba(255,255,255,0.03)",
            },
          }}
        >
          <RefreshOutlined sx={{ fontSize: 18 }} />
        </IconButton>
      </Box>

      <TextField
        size="small"
        placeholder="Search models..."
        value={modelSearch}
        onChange={(e) => onSearchChange(e.target.value)}
        fullWidth
        slotProps={{
          input: {
            startAdornment: (
              <InputAdornment position="start">
                <SearchOutlined sx={{ fontSize: 16, color: "#52525B" }} />
              </InputAdornment>
            ),
            sx: {
              fontFamily: "'Outfit', sans-serif",
              fontSize: "0.8125rem",
              color: "#FAFAFA",
              bgcolor: "rgba(255,255,255,0.03)",
              borderRadius: "8px",
              "& fieldset": {
                borderColor: "#27272A",
                transition: "border-color 0.15s ease",
              },
              "&:hover fieldset": {
                borderColor: "#3F3F46 !important",
              },
              "&.Mui-focused fieldset": {
                borderColor: "#F97316 !important",
                borderWidth: "1px !important",
              },
              "& input::placeholder": {
                color: "#52525B",
                opacity: 1,
              },
            },
          },
        }}
        sx={{ mb: 2.5, "& .MuiOutlinedInput-root": { height: 40 } }}
      />
    </>
  );
}
