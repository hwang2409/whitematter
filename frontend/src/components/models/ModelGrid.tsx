"use client";
import React from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import TableSortLabel from "@mui/material/TableSortLabel";
import Collapse from "@mui/material/Collapse";
import IconButton from "@mui/material/IconButton";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import * as api from "@/api";
import ModelStatusBadge from "./ModelStatusBadge";
import { SortColumn, SortDirection, formatModelName, formatDate, getModelLoss } from "./useModelList";

const COLUMNS: { id: SortColumn; label: string }[] = [
  { id: "name", label: "Name" },
  { id: "architecture", label: "Architecture" },
  { id: "status", label: "Status" },
  { id: "accuracy", label: "Accuracy" },
  { id: "loss", label: "Loss" },
  { id: "created", label: "Created" },
];

const headerCellSx = {
  fontFamily: "'Outfit', sans-serif",
  fontSize: "0.625rem",
  fontWeight: 600,
  textTransform: "uppercase" as const,
  letterSpacing: "0.08em",
  color: "#52525B",
  borderBottom: "1px solid #27272A",
  py: 1.25,
  px: 1.5,
  whiteSpace: "nowrap" as const,
};

const bodyCellSx = {
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: "0.8125rem",
  color: "#FAFAFA",
  borderBottom: "1px solid rgba(39,39,42,0.5)",
  py: 1.25,
  px: 1.5,
};

interface ModelGridProps {
  models: api.Model[];
  expandedModelId: string | null;
  sortColumn: SortColumn;
  sortDirection: SortDirection;
  onSortClick: (column: SortColumn) => void;
  onRowClick: (model: api.Model) => void;
  renderExpandedContent: (model: api.Model) => React.ReactNode;
}

export default function ModelGrid({
  models,
  expandedModelId,
  sortColumn,
  sortDirection,
  onSortClick,
  onRowClick,
  renderExpandedContent,
}: ModelGridProps) {
  if (models.length === 0) {
    return null;
  }

  return (
    <Box
      sx={{
        overflowX: "auto",
        borderRadius: "12px",
        border: "1px solid #27272A",
        bgcolor: "#18181B",
      }}
    >
      <Table size="small" sx={{ "& .MuiTableCell-root": { border: 0 } }}>
        <TableHead>
          <TableRow>
            <TableCell sx={{ ...headerCellSx, width: 44, px: 0.5 }} />
            {COLUMNS.map((col) => (
              <TableCell
                key={col.id}
                sortDirection={sortColumn === col.id ? sortDirection : false}
                sx={headerCellSx}
              >
                <TableSortLabel
                  active={sortColumn === col.id}
                  direction={sortColumn === col.id ? sortDirection : "asc"}
                  onClick={() => onSortClick(col.id)}
                  sx={{
                    color: "#52525B !important",
                    "&:hover": { color: "#A1A1AA !important" },
                    "&.Mui-active": { color: "#A1A1AA !important" },
                    "& .MuiTableSortLabel-icon": {
                      color: "#52525B !important",
                      fontSize: "0.875rem",
                    },
                    "&.Mui-active .MuiTableSortLabel-icon": {
                      color: "#A1A1AA !important",
                    },
                  }}
                >
                  {col.label}
                </TableSortLabel>
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {models.map((model) => {
            const isExpanded = expandedModelId === model.id;
            const loss = getModelLoss(model);
            return (
              <React.Fragment key={model.id}>
                <TableRow
                  hover
                  onClick={() => onRowClick(model)}
                  sx={{
                    cursor: "pointer",
                    transition: "background-color 0.1s ease",
                    "&:hover": {
                      bgcolor: "rgba(255,255,255,0.02) !important",
                    },
                    "& > *": {
                      borderBottom: isExpanded
                        ? "none !important"
                        : "1px solid rgba(39,39,42,0.5) !important",
                    },
                  }}
                >
                  <TableCell sx={{ ...bodyCellSx, width: 44, px: 0.5, borderBottom: "none" }}>
                    <IconButton
                      size="small"
                      aria-label="expand row"
                      sx={{
                        color: "#52525B",
                        transition: "color 0.15s ease, transform 0.2s ease",
                        transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
                        "&:hover": { color: "#A1A1AA" },
                      }}
                    >
                      <KeyboardArrowDownIcon sx={{ fontSize: 18 }} />
                    </IconButton>
                  </TableCell>
                  <TableCell sx={bodyCellSx}>
                    <Typography
                      variant="body2"
                      sx={{
                        fontFamily: "'Outfit', sans-serif",
                        fontWeight: 500,
                        fontSize: "0.875rem",
                        color: "#FAFAFA",
                      }}
                    >
                      {formatModelName(model.name)}
                    </Typography>
                  </TableCell>
                  <TableCell sx={bodyCellSx}>
                    <Typography
                      variant="body2"
                      sx={{
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: "0.8125rem",
                        color: "#A1A1AA",
                      }}
                    >
                      {model.architecture}
                    </Typography>
                  </TableCell>
                  <TableCell sx={bodyCellSx}>
                    <ModelStatusBadge status={model.status} />
                  </TableCell>
                  <TableCell sx={bodyCellSx}>
                    <Typography
                      variant="body2"
                      sx={{
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: "0.8125rem",
                        color: "#FAFAFA",
                      }}
                    >
                      {model.best_accuracy.toFixed(2)}%
                    </Typography>
                  </TableCell>
                  <TableCell sx={bodyCellSx}>
                    <Typography
                      variant="body2"
                      sx={{
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: "0.8125rem",
                        color: "#A1A1AA",
                      }}
                    >
                      {loss != null ? loss.toFixed(4) : "\u2014"}
                    </Typography>
                  </TableCell>
                  <TableCell sx={bodyCellSx}>
                    <Typography
                      variant="body2"
                      sx={{
                        fontFamily: "'Outfit', sans-serif",
                        fontSize: "0.8125rem",
                        color: "#52525B",
                      }}
                    >
                      {formatDate(model.created_at)}
                    </Typography>
                  </TableCell>
                </TableRow>
                <TableRow key={`${model.id}-expand`}>
                  <TableCell
                    sx={{
                      py: 0,
                      px: 0,
                      borderBottom: isExpanded
                        ? "1px solid rgba(39,39,42,0.5) !important"
                        : "none !important",
                    }}
                    colSpan={7}
                  >
                    <Collapse in={isExpanded} timeout={300} unmountOnExit>
                      {renderExpandedContent(model)}
                    </Collapse>
                  </TableCell>
                </TableRow>
              </React.Fragment>
            );
          })}
        </TableBody>
      </Table>
    </Box>
  );
}
