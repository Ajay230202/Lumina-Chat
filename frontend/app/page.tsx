"use client";

import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, 
  Upload, 
  Image as ImageIcon, 
  FileText, 
  Music, 
  Video, 
  Sparkles,
  FileCheck,
  Loader2,
  FolderOpen,
  Info,
  Pencil
} from 'lucide-react';
import { Message, Source } from '../lib/types';
import { streamChat, uploadFile, getIngestStatus, getDocuments, createSession, getSessionHistory } from '../lib/api';

interface TableData {
  headers: string[];
  rows: string[][];
}

function parseMessageContent(content: string): React.ReactNode[] {
  if (!content) return [];
  
  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];
  let currentTable: TableData | null = null;
  let textBuffer: string[] = [];

  const flushTextBuffer = (key: string) => {
    if (textBuffer.length > 0) {
      elements.push(
        <p key={key} className="text-sm leading-relaxed whitespace-pre-wrap mb-4 last:mb-0">
          {textBuffer.join('\n')}
        </p>
      );
      textBuffer = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    // Check if it's a table row
    if (line.startsWith('|') && line.endsWith('|')) {
      // Split by '|' and clean up cells
      const cells = line.split('|').map(c => c.trim()).slice(1, -1);
      
      if (currentTable === null) {
        // Look ahead to check if the next line is a divider
        const nextLine = lines[i + 1]?.trim();
        if (nextLine && nextLine.startsWith('|') && nextLine.includes('---')) {
          // Start a new table
          currentTable = {
            headers: cells,
            rows: []
          };
          // Skip the divider line
          i++;
          flushTextBuffer(`text-pre-table-${i}`);
        } else {
          // Not a table header (just a line with pipes), treat as normal text
          textBuffer.push(lines[i]);
        }
      } else {
        // Add row to existing table
        currentTable.rows.push(cells);
      }
    } else {
      if (currentTable !== null) {
        // Table ended, render it
        elements.push(renderVisualTable(currentTable, `table-${i}`));
        currentTable = null;
      }
      textBuffer.push(lines[i]);
    }
  }

  if (currentTable !== null) {
    elements.push(renderVisualTable(currentTable, `table-end`));
  }
  flushTextBuffer(`text-end`);

  return elements;
}

function renderVisualTable(table: TableData, key: string): React.ReactNode {
  return (
    <div key={key} className="my-5 overflow-x-auto rounded-xl border border-[#E8E2D9] shadow-sm bg-white">
      <table className="w-full text-left border-collapse text-xs">
        <thead>
          <tr className="bg-[#F4F0EA] border-b border-[#E8E2D9] text-[#2C2621]">
            {table.headers.map((h, i) => (
              <th key={i} className="p-3 font-bold uppercase tracking-wider text-[10px]">
                {h.replace(/\*\*/g, '')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-[#E8E2D9]">
          {table.rows.map((row, rowIndex) => (
            <tr key={rowIndex} className="hover:bg-[#FBF9F6] transition-colors text-[#2C2621]">
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className="p-3 whitespace-pre-wrap font-medium">
                  {cell.replace(/\*\*/g, '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function compressImage(file: File, maxWidth: number, maxHeight: number, quality: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        let width = img.width;
        let height = img.height;

        if (width > height) {
          if (width > maxWidth) {
            height = Math.round((height * maxWidth) / width);
            width = maxWidth;
          }
        } else {
          if (height > maxHeight) {
            width = Math.round((width * maxHeight) / height);
            height = maxHeight;
          }
        }

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error("Canvas context is null"));
          return;
        }

        // Fill background with solid white to preserve transparent PNG visibility in JPEG conversion
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, width, height);

        ctx.drawImage(img, 0, 0, width, height);
        const dataUrl = canvas.toDataURL('image/jpeg', quality);
        resolve(dataUrl);
      };
      img.onerror = () => reject(new Error("Failed to load image"));
      img.src = e.target?.result as string;
    };
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [dept, setDept] = useState('General');
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [imageFileName, setImageFileName] = useState<string | null>(null);
  const [documents, setDocuments] = useState<any[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingContent, setEditingContent] = useState<string>('');
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Fetch documents list from db
  const fetchDocs = async () => {
    try {
      const docs = await getDocuments();
      setDocuments(docs);
    } catch (err) {
      console.error("Error fetching documents:", err);
    }
  };

  // Auto-scroll chat and load docs on mount
  useEffect(() => {
    const initSession = async () => {
      try {
        const savedId = localStorage.getItem('lumina_session_id');
        if (savedId) {
          setSessionId(savedId);
          const history = await getSessionHistory(savedId);
          if (history && history.length > 0) {
            const uiMessages = history.map((msg: any) => ({
              id: msg.id || `${Date.now()}-${Math.random()}`,
              role: msg.role,
              content: msg.content,
              image_b64: msg.image_b64 || undefined,
              sources: msg.source_chunks || []
            }));
            setMessages(uiMessages);
          }
        } else {
          const session = await createSession();
          localStorage.setItem('lumina_session_id', session.id);
          setSessionId(session.id);
        }
      } catch (err) {
        console.error("Failed to initialize session:", err);
      }
    };
    initSession();
    fetchDocs();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle image attachment at query time
  const handleImageChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFileName(file.name);
      try {
        const compressedDataUrl = await compressImage(file, 1024, 1024, 0.75);
        setImageB64(compressedDataUrl);
      } catch (err) {
        console.error("Error compressing image:", err);
        const reader = new FileReader();
        reader.onloadend = () => {
          setImageB64(reader.result as string);
        };
        reader.readAsDataURL(file);
      }
    }
  };

  // Handle document upload ingestion
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadStatus("Uploading file...");
    fetchDocs(); // Refresh documents list to show "pending" state instantly
    
    try {
      const result = await uploadFile(file, dept);
      setUploadStatus("Ingesting & parsing...");
      fetchDocs(); // Refresh list to update ID status
      
      // Poll status
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await getIngestStatus(result.doc_id);
          if (statusRes.status === 'ready') {
            clearInterval(pollInterval);
            setUploadStatus("Ingested successfully!");
            setIsUploading(false);
            fetchDocs(); // Refresh to show completed state
            setTimeout(() => setUploadStatus(null), 5000);
          } else if (statusRes.status === 'failed') {
            clearInterval(pollInterval);
            setUploadStatus("Ingestion failed.");
            setIsUploading(false);
            fetchDocs(); // Refresh status
          } else {
            setUploadStatus(`Processing: ${statusRes.status}...`);
          }
        } catch (pollErr) {
          clearInterval(pollInterval);
          setUploadStatus("In progress (status check failed).");
          setIsUploading(false);
          fetchDocs();
        }
      }, 3000);

    } catch (err: any) {
      setUploadStatus(`Error: ${err.message || 'Failed to upload'}`);
      setIsUploading(false);
      fetchDocs();
      setTimeout(() => setUploadStatus(null), 6000);
    }
  };

  // Submit query
  const handleSend = async () => {
    if (!input.trim() && !imageB64) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      image_b64: imageB64 || undefined
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setImageB64(null);
    setImageFileName(null);
    setIsGenerating(true);

    const assistantMsgId = (Date.now() + 1).toString();
    const newAssistantMessage: Message = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      sources: []
    };

    setMessages(prev => [...prev, newAssistantMessage]);

    // Create abort controller for stop generation
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await streamChat(
        userMessage.content,
        messages.concat(userMessage),
        userMessage.image_b64,
        // onChunk
        (textChunk) => {
          setMessages(prev => prev.map(msg => {
            if (msg.id === assistantMsgId) {
              return { ...msg, content: msg.content + textChunk };
            }
            return msg;
          }));
        },
        // onSources
        (citations) => {
          setMessages(prev => prev.map(msg => {
            if (msg.id === assistantMsgId) {
              return { ...msg, sources: citations };
            }
            return msg;
          }));
        },
        // onError
        (errorMessage) => {
          if (controller.signal.aborted) return;
          setMessages(prev => prev.map(msg => {
            if (msg.id === assistantMsgId) {
              return { ...msg, content: msg.content + `\n\n*(Error: ${errorMessage})*` };
            }
            return msg;
          }));
          setIsGenerating(false);
        },
        sessionId || undefined,
        controller.signal
      );
    } catch (e: any) {
      if (e.name === 'AbortError') {
        console.log("Chat generation stopped by user");
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsGenerating(false);
    }
  };

  const resubmitChat = async (historyBeforeMsg: Message[], userMsg: Message) => {
    setIsGenerating(true);
    const updatedMessages = [...historyBeforeMsg, userMsg];
    setMessages(updatedMessages);
    
    const assistantMsgId = (Date.now() + 1).toString();
    const newAssistantMessage: Message = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      sources: []
    };
    
    setMessages(prev => [...prev, newAssistantMessage]);
    
    const controller = new AbortController();
    abortControllerRef.current = controller;
    
    try {
      await streamChat(
        userMsg.content,
        historyBeforeMsg.concat(userMsg),
        userMsg.image_b64,
        (textChunk) => {
          setMessages(prev => prev.map(msg => {
            if (msg.id === assistantMsgId) {
              return { ...msg, content: msg.content + textChunk };
            }
            return msg;
          }));
        },
        (citations) => {
          setMessages(prev => prev.map(msg => {
            if (msg.id === assistantMsgId) {
              return { ...msg, sources: citations };
            }
            return msg;
          }));
        },
        (errorMessage) => {
          if (controller.signal.aborted) return;
          setMessages(prev => prev.map(msg => {
            if (msg.id === assistantMsgId) {
              return { ...msg, content: msg.content + `\n\n*(Error: ${errorMessage})*` };
            }
            return msg;
          }));
          setIsGenerating(false);
        },
        sessionId || undefined,
        controller.signal
      );
    } catch (e: any) {
      if (e.name === 'AbortError') {
        console.log("Chat generation stopped by user");
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  const handleSaveEdit = (messageId: string) => {
    const index = messages.findIndex(m => m.id === messageId);
    if (index !== -1) {
      const historyBeforeMsg = messages.slice(0, index);
      const userMsg = {
        ...messages[index],
        content: editingContent
      };
      setEditingMessageId(null);
      resubmitChat(historyBeforeMsg, userMsg);
    }
  };

  const getHeaderLabel = (d: string) => {
    switch (d) {
      case 'HR': return "Lumina HR Intelligence Online";
      case 'Finance': return "Lumina Fiscal Analyst Online";
      case 'Policy': return "Lumina Corporate Compliance Engine Active";
      case 'Legal': return "Lumina Legal Counsel Engine Active";
      default: return "Lumina Neural Engine Active";
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[#FBF9F6] text-[#2C2621]">
      {/* LEFT SIDEBAR - Ingestion Console */}
      <aside className="w-80 bg-[#F4F0EA] border-r border-[#E8E2D9] flex flex-col justify-between h-full">
        <div className="p-6 overflow-y-auto flex-1 space-y-6">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-tr from-[#c05a3e] to-[#a84c30] p-2.5 rounded-xl text-white shadow-md">
              <FolderOpen size={20} />
            </div>
            <div>
              <h1 className="font-semibold text-base text-[#2C2621]">Lumina RAG</h1>
              <span className="text-xs text-[#6E645E]">Multimodal Agentic RAG</span>
            </div>
          </div>

          {/* Department Selector */}
          <div>
            <label className="block text-[10px] font-bold text-[#6E645E] uppercase tracking-wider mb-2">
              Target Department
            </label>
            <select
              value={dept}
              onChange={(e) => setDept(e.target.value)}
              className="w-full bg-white border border-[#E8E2D9] rounded-lg px-3 py-2 text-sm text-[#2C2621] focus:outline-none focus:border-[#c05a3e] cursor-pointer transition-colors shadow-sm"
            >
              <option value="General">General</option>
              <option value="HR">HR & Personnel</option>
              <option value="Finance">Finance & Tax</option>
              <option value="Policy">Company Policy</option>
              <option value="Legal">Legal & Contracts</option>
            </select>
          </div>

          {/* Ingestion Upload Card */}
          <div className="bg-white border border-[#E8E2D9] rounded-xl p-5 text-center relative overflow-hidden shadow-sm hover:border-[#c05a3e] transition-colors">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
              accept=".pdf,.docx,.pptx,.mp3,.wav,.m4a,.mp4,.avi,.mov"
            />
            
            <div className="flex flex-col items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-[#F4F0EA] flex items-center justify-center text-[#c05a3e] border border-[#E8E2D9] cursor-pointer hover:bg-white hover:border-[#c05a3e] transition-colors"
                   onClick={() => fileInputRef.current?.click()}>
                <Upload size={18} />
              </div>
              <div>
                <button 
                  onClick={() => fileInputRef.current?.click()}
                  className="text-sm font-semibold text-[#2C2621] hover:text-[#c05a3e] transition-colors"
                >
                  Upload knowledge source
                </button>
                <p className="text-[11px] text-[#6E645E] mt-1">PDF, Office, Audio, Video</p>
              </div>
            </div>

            {uploadStatus && (
              <div className="mt-4 pt-3 border-t border-[#E8E2D9] flex items-center justify-center gap-2 text-xs text-[#c05a3e] font-bold bg-[#FBF9F6] p-2 rounded-lg">
                {isUploading && <Loader2 size={12} className="animate-spin text-[#c05a3e]" />}
                <span>{uploadStatus}</span>
              </div>
            )}
          </div>

          {/* Knowledge Base Section */}
          <div className="border-t border-[#E8E2D9] pt-5">
            <h3 className="block text-[10px] font-bold text-[#6E645E] uppercase tracking-wider mb-3">
              Knowledge Base ({documents.length})
            </h3>
            {documents.length === 0 ? (
              <p className="text-xs text-[#9A8F87] italic">No sources uploaded yet.</p>
            ) : (
              <div className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
                {documents.map((doc) => (
                  <div key={doc.id} className="bg-white border border-[#E8E2D9] rounded-xl p-2.5 flex items-center justify-between shadow-sm">
                    <div className="flex items-center gap-2 overflow-hidden">
                      <div className="shrink-0 p-1.5 rounded-lg bg-[#F4F0EA]">
                        {doc.file_type === 'pdf' && <FileText size={15} className="text-[#c05a3e]" />}
                        {(doc.file_type === 'docx' || doc.file_type === 'pptx') && <FileCheck size={15} className="text-amber-600" />}
                        {(doc.file_type === 'mp3' || doc.file_type === 'wav' || doc.file_type === 'm4a') && <Music size={15} className="text-purple-600" />}
                        {(doc.file_type === 'mp4' || doc.file_type === 'avi' || doc.file_type === 'mov') && <Video size={15} className="text-rose-600" />}
                      </div>
                      <div className="overflow-hidden">
                        <p className="text-xs font-semibold text-[#2C2621] truncate" title={doc.filename}>
                          {doc.filename}
                        </p>
                        <span className="text-[9px] text-[#6E645E] font-medium tracking-wide bg-[#F4F0EA] px-1 rounded uppercase">{doc.dept}</span>
                      </div>
                    </div>
                    <div className="shrink-0 pl-1">
                      {doc.status === 'ready' && (
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 block" title="Ready"></span>
                      )}
                      {doc.status === 'pending' || doc.status === 'processing' ? (
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-500 block animate-pulse" title="Processing"></span>
                      ) : null}
                      {doc.status === 'failed' && (
                        <span className="w-1.5 h-1.5 rounded-full bg-rose-500 block" title="Failed"></span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Sidebar Footer info */}
        <div className="p-6 border-t border-[#E8E2D9]">
          <div className="flex gap-3 text-xs text-[#6E645E]">
            <Info size={16} className="text-[#c05a3e] shrink-0" />
            <p>Lumina processes text, charts, audio, and video to construct a unified brain for your enterprise.</p>
          </div>
        </div>
      </aside>

      {/* RIGHT MAIN AREA - Chat Interface */}
      <main className="flex-1 flex flex-col bg-[#FBF9F6]">
        {/* Top Header */}
        <header className="h-16 border-b border-[#E8E2D9] bg-[#F4F0EA] px-8 flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="text-xs font-semibold text-[#2C2621] tracking-wide uppercase">{getHeaderLabel(dept)}</span>
          </div>
          <div className="text-xs text-[#6E645E] font-medium tracking-wide">
            Multi-Agent Hybrid Retrieval & Cognitive Synthesis Loop
          </div>
        </header>

        {/* Message Area */}
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto space-y-4">
              <div className="w-16 h-16 rounded-full bg-[#F4F0EA] flex items-center justify-center text-[#c05a3e] border border-[#E8E2D9] shadow-inner animate-pulse">
                <Sparkles size={24} />
              </div>
              <h2 className="text-lg font-bold text-[#2C2621] font-serif">Consult Lumina Knowledge Engine</h2>
              <p className="text-sm text-[#6E645E] leading-relaxed">
                Upload text, sheets, voice notes, or charts and ask Lumina to analyze, synthesize, and visualize answers for you.
              </p>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-2xl rounded-2xl p-5 ${
                  msg.role === 'user' 
                    ? 'bg-[#4E3D30] text-white rounded-br-none shadow-sm' 
                    : 'bg-white border border-[#E8E2D9] text-[#2C2621] rounded-bl-none shadow-sm'
                }`}>
                  
                  {/* Attached Image preview inside message bubble */}
                  {msg.image_b64 && (
                    <div className="mb-3 overflow-hidden rounded-lg border border-[#E8E2D9] max-h-60 bg-[#F4F0EA]">
                      <img 
                        src={msg.image_b64.startsWith('data:') ? msg.image_b64 : `data:image/jpeg;base64,${msg.image_b64}`} 
                        alt="Query Attachment" 
                        className="object-contain w-full h-full"
                      />
                    </div>
                  )}

                  {/* Message Text Content */}
                  <div className="text-sm leading-relaxed relative group">
                    {msg.role === 'user' && editingMessageId === msg.id ? (
                      <div className="space-y-2 min-w-[240px]">
                        <textarea
                          value={editingContent}
                          onChange={(e) => setEditingContent(e.target.value)}
                          className="w-full bg-[#3D2E24] text-white rounded-lg p-2 text-xs focus:outline-none border border-[#c05a3e] resize-none"
                          rows={3}
                        />
                        <div className="flex justify-end gap-2">
                          <button
                            onClick={() => setEditingMessageId(null)}
                            className="text-[10px] uppercase font-bold tracking-wider px-2 py-1 rounded bg-[#5A4A3F] text-[#E8E2D9] hover:bg-[#6E5A4D] transition-colors"
                          >
                            Cancel
                          </button>
                          <button
                            onClick={() => handleSaveEdit(msg.id)}
                            className="text-[10px] uppercase font-bold tracking-wider px-2 py-1 rounded bg-[#c05a3e] text-white hover:bg-[#a84c30] transition-colors"
                          >
                            Save & Send
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        {msg.content ? (
                          parseMessageContent(msg.content)
                        ) : (
                          isGenerating && msg.role === 'assistant' ? (
                            <div className="flex items-center gap-2 text-[#c05a3e] font-semibold animate-pulse py-1">
                              <Loader2 size={14} className="animate-spin text-[#c05a3e]" />
                              <span>Generating response...</span>
                            </div>
                          ) : null
                        )}
                        
                        {/* Pencil Edit button for user messages */}
                        {msg.role === 'user' && !editingMessageId && !isGenerating && (
                          <button
                            onClick={() => {
                              setEditingMessageId(msg.id);
                              setEditingContent(msg.content);
                            }}
                            className="absolute -top-7 -right-2 p-1.5 rounded-full bg-[#5C4B3E] hover:bg-[#c05a3e] text-white opacity-0 group-hover:opacity-100 transition-all shadow-sm"
                            title="Edit query"
                          >
                            <Pencil size={10} />
                          </button>
                        )}
                      </>
                    )}
                  </div>

                  {/* Message Citations Sources Accordion */}
                  {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-[#E8E2D9] space-y-2">
                      <span className="text-[10px] uppercase font-bold text-[#6E645E] tracking-wider">
                        Sources cited
                      </span>
                      <div className="grid grid-cols-1 gap-2">
                        {msg.sources.map((src, i) => (
                          <details key={src.chunk_id} className="text-xs bg-[#FBF9F6] border border-[#E8E2D9] rounded-lg p-2.5 cursor-pointer group hover:bg-[#F4F0EA] transition-colors">
                            <summary className="font-semibold text-[#c05a3e] flex items-center justify-between hover:text-[#a84c30] transition-colors">
                              <div className="flex items-center gap-2">
                                <span className="bg-[#E8E2D9] text-[#2C2621] px-1.5 py-0.5 rounded text-[10px] font-bold">
                                  [Source {i+1}]
                                </span>
                                {src.modality === 'text' && <FileText size={12} className="text-[#6E645E]" />}
                                {src.modality === 'image' && <ImageIcon size={12} className="text-emerald-600" />}
                                {src.modality === 'table' && <FileCheck size={12} className="text-amber-600" />}
                                {src.modality === 'audio_transcript' && <Music size={12} className="text-purple-600" />}
                                {src.modality === 'video_frame' && <Video size={12} className="text-rose-600" />}
                                <span className="capitalize">{src.modality.replace('_', ' ')}</span>
                                {src.page_num && <span className="text-[#6E645E]">(Page {src.page_num})</span>}
                              </div>
                              {src.score && (
                                <span className="text-[#6E645E] text-[10px] font-medium">
                                  Score: {src.score.toFixed(3)}
                                </span>
                              )}
                            </summary>
                            <p className="mt-2 text-[#6E645E] border-l-2 border-[#E8E2D9] pl-2 py-1 leading-normal text-[11px] whitespace-pre-wrap select-text font-sans">
                              {src.text_repr}
                            </p>
                          </details>
                        ))}
                      </div>
                    </div>
                  )}

                </div>
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Bar Area */}
        <footer className="p-6 border-t border-[#E8E2D9] bg-[#F4F0EA]">
          <div className="max-w-3xl mx-auto">
            {/* Attached Image preview banner before sending */}
            {imageFileName && (
              <div className="mb-2 p-2 bg-white border border-[#E8E2D9] rounded-lg flex items-center justify-between text-xs text-[#c05a3e] font-semibold">
                <span className="flex items-center gap-2">
                  <ImageIcon size={14} />
                  {imageFileName}
                </span>
                <button 
                  onClick={() => { setImageB64(null); setImageFileName(null); }}
                  className="text-[#6E645E] hover:text-[#2C2621]"
                >
                  Remove
                </button>
              </div>
            )}

            <div className="relative flex items-center bg-white border border-[#E8E2D9] rounded-xl px-4 py-2 focus-within:border-[#c05a3e] focus-within:ring-1 focus-within:ring-[#c05a3e] transition-all shadow-sm">
              <input
                type="text"
                placeholder="Ask a question..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                disabled={isGenerating}
                className="flex-1 bg-transparent text-[#2C2621] text-sm focus:outline-none placeholder-[#9A8F87] pr-12"
              />

              <div className="flex items-center gap-2">
                <input
                  type="file"
                  ref={imageInputRef}
                  onChange={handleImageChange}
                  className="hidden"
                  accept="image/*"
                />
                
                <button
                  type="button"
                  onClick={() => imageInputRef.current?.click()}
                  disabled={isGenerating}
                  className="text-[#6E645E] hover:text-[#2C2621] transition-colors p-1.5 rounded-lg hover:bg-[#F4F0EA]"
                  title="Attach image"
                >
                  <ImageIcon size={18} />
                </button>

                {isGenerating ? (
                  <button
                    type="button"
                    onClick={handleStop}
                    className="bg-rose-600 hover:bg-rose-500 text-white rounded-lg p-2.5 transition-colors animate-pulse flex items-center justify-center"
                    title="Stop generation"
                  >
                    <span className="w-3 h-3 bg-white rounded-xs block"></span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleSend}
                    disabled={!input.trim() && !imageB64}
                    className="bg-[#c05a3e] hover:bg-[#a84c30] text-white rounded-lg p-2.5 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
                    title="Send message"
                  >
                    <Send size={16} />
                  </button>
                )}
              </div>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}
