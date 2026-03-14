"use client";
import { use } from "react";
import ChatPage from "@/views/ChatPage";

export default function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  return <ChatPage conversationId={id} />;
}
