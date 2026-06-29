import './globals.css';
import React from 'react';

export const metadata = {
  title: '🧠 Multimodal Agentic RAG Chatbot',
  description: 'Powered by NVIDIA NIM, LangGraph, Qdrant, and Supabase',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen flex flex-col bg-[#0d1117]">
        {children}
      </body>
    </html>
  );
}
