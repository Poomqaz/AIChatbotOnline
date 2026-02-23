/**
 * ===============================================
 * Document Loader, Embedding & PGVector API
 * ===============================================
 * 
 * ฟีเจอร์หลัก:
 * - โหลดและประมวลผลเอกสารจากโฟลเดอร์ data/
 * - แปลงเอกสารเป็น embeddings ด้วย OpenAI
 * - เก็บใน Supabase Vector Store (pgvector)
 * - รองรับไฟล์ .txt และ .csv
 * - Text splitting สำหรับ chunk ขนาดเหมาะสม
 * - ป้องกันข้อมูลซ้ำซ้อนด้วยการลบข้อมูลเก่าก่อนโหลดใหม่
 * 
 * API Endpoints:
 * - GET: โหลดเอกสารและสร้าง embeddings (ลบข้อมูลเก่าก่อนโหลดใหม่)
 * - POST: ค้นหาเอกสารที่คล้ายกันด้วย similarity search
 * - PUT: ดูสถิติข้อมูลใน vector store
 * - DELETE: ลบข้อมูลทั้งหมดใน vector store
 */

import { NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/server";

// LangChain & AI SDK Imports
import { DirectoryLoader } from "@langchain/classic/document_loaders/fs/directory";
import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase"
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { CacheBackedEmbeddings } from "@langchain/classic/embeddings/cache_backed";
import { InMemoryStore } from "@langchain/core/stores"

export const dynamic = 'force-dynamic'
export const maxDuration = 60 // เพิ่มเวลาสำหรับการประมวลผล

/**
 * GET API: โหลดเอกสาร สร้าง embeddings และเก็บใน vector store
 */
export async function GET() {
  try {
    console.log("🔄 เริ่มโหลดเอกสารจากโฟลเดอร์ data/...")
    
    // ===============================================
    // Step 0: ตรวจสอบและลบข้อมูลเก่า - Clean Existing Data
    // ===============================================
    const supabase = await createClient();
    
    const { count: existingCount } = await supabase
      .from('documents')
      .select('*', { count: 'exact', head: true });

    if (existingCount && existingCount > 0) {
      console.log(`🗑️ พบข้อมูลเก่า ${existingCount} records - ลบข้อมูลเก่าก่อน...`);
      const { error: deleteError } = await supabase
        .from('documents')
        .delete()
        .neq('id', 0);

      if (deleteError) throw new Error(`ไม่สามารถลบข้อมูลเก่าได้: ${deleteError.message}`);
      console.log(`✅ ลบข้อมูลเก่าสำเร็จ`);
    }
    
    // ===============================================
    // Step 1: โหลดเอกสารจากไดเร็กทอรี
    // ===============================================
    const rawDocs = await new DirectoryLoader("./data", {
        ".txt": (path) => new TextLoader(path),
        ".csv": (path) => new CSVLoader(path, { column: undefined, separator: "," }),
    }).load();

    if (rawDocs.length === 0) {
      return NextResponse.json({ error: "ไม่พบเอกสารในโฟลเดอร์ data/" }, { status: 400 })
    }

    // ===============================================
    // Step 2: แยกเอกสาร (Chunking)
    // ===============================================
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,    // ปรับขนาดให้เหมาะสม (Google รองรับ context ได้เยอะ)
        chunkOverlap: 200,
        separators: ["\n\n", "\n", ",", " "],
    });

    const chunks = await splitter.splitDocuments(rawDocs);
    console.log(`✂️ แยกเอกสารเป็น ${chunks.length} ชิ้น`)

    // ===============================================
    // Step 3: เตรียม Embeddings และ Vector Store
    // ===============================================
    const baseEmbeddings = new GoogleGenerativeAIEmbeddings({ 
      model: process.env.GOOGLE_EMBEDDING_MODEL_NAME || 'text-embedding-004',
      // taskType: TaskType.RETRIEVAL_DOCUMENT, // บางเวอร์ชั่นต้องระบุ TaskType
    });

    const cacheStore = new InMemoryStore();
    const embeddings = CacheBackedEmbeddings.fromBytesStore(
      baseEmbeddings,
      cacheStore,
      { namespace: "document_embeddings" }
    );

    const vectorStore = new SupabaseVectorStore(embeddings, {
        client: supabase,
        tableName: 'documents',
        queryName: 'match_documents' 
    });

    // ===============================================
    // Step 4: เพิ่ม Metadata
    // ===============================================
    const chunksWithMetadata = chunks.map((chunk, index) => {
      const source = chunk.metadata.source || 'unknown'
      const filename = source.split('/').pop() || source.split('\\').pop() || 'unknown'
      
      return {
        ...chunk,
        metadata: {
          ...chunk.metadata,
          filename,
          chunk_index: index,
          chunk_size: chunk.pageContent.length,
          timestamp: new Date().toISOString(),
          type: filename.endsWith('.csv') ? 'csv' : 'text'
        }
      }
    })

    // ===============================================
    // Step 5: สร้าง Embeddings แบบ Batch (แก้ไขจุดที่ Error)
    // ===============================================
    console.log("🔮 กำลังสร้าง embeddings และบันทึก (Batch Processing)...")
    
    // กำหนดขนาด Batch เพื่อป้องกัน Timeout / Rate Limit
    const BATCH_SIZE = 50; 
    let processedCount = 0;

    for (let i = 0; i < chunksWithMetadata.length; i += BATCH_SIZE) {
      const batch = chunksWithMetadata.slice(i, i + BATCH_SIZE);
      console.log(`⚡ กำลังประมวลผล Batch ${i + 1} - ${Math.min(i + BATCH_SIZE, chunksWithMetadata.length)} จาก ${chunksWithMetadata.length}`);
      
      try {
        await vectorStore.addDocuments(batch);
        processedCount += batch.length;
        
        // (Optional) พัก 0.5 วินาที เพื่อลดภาระ API
        await new Promise(resolve => setTimeout(resolve, 500)); 
        
      } catch (batchError) {
        console.error(`❌ Error ใน Batch ที่เริ่มต้น index ${i}:`, batchError);
        // สามารถเลือกที่จะ throw error เลย หรือจะข้าม batch ที่เสียไปก็ได้
        throw batchError; 
      }
    }
    
    console.log(`✅ สำเร็จ! เก็บข้อมูลทั้งหมด ${processedCount} chunks แล้ว`)

    // ===============================================
    // Step 6: สร้างสถิติ
    // ===============================================
    const stats = {
      previous_records: existingCount || 0,
      new_records: processedCount,
      total_documents: rawDocs.length,
      embedding_model: process.env.GOOGLE_EMBEDDING_MODEL_NAME || 'text-embedding-004',
      timestamp: new Date().toISOString()
    }

    return NextResponse.json({ 
      message: `สำเร็จ! ประมวลผล ${processedCount} chunks`,
      stats,
      success: true
    })

  } catch (error) {
    console.error('❌ Error ใหญ่ในการประมวลผล:', error)
    
    return NextResponse.json({ 
      error: 'เกิดข้อผิดพลาดในการประมวลผลเอกสาร',
      details: error instanceof Error ? error.message : 'Unknown error',
      suggestion: 'ลองเช็คขนาด Vector Column ใน Supabase (Google ใช้ 768, OpenAI ใช้ 1536)',
      success: false
    }, { status: 500 })
  }
}

/**
 * POST API: ค้นหาเอกสารที่คล้ายกันใน vector store
 */
export async function POST(req: NextRequest) {
  try {
    const { query, limit = 5 } = await req.json()
    
    if (!query) {
      return NextResponse.json({ 
        error: "กรุณาระบุ query สำหรับการค้นหา" 
      }, { status: 400 })
    }

    console.log(`🔍 ค้นหา: "${query}"`)
    console.log("⚡ ใช้ CacheBackedEmbeddings สำหรับการค้นหา")

    // ===============================================
    // Setup Vector Store สำหรับการค้นหา
    // ===============================================
    const supabase = await createClient();
    
    const baseEmbeddings = new GoogleGenerativeAIEmbeddings({ 
      model: process.env.GOOGLE_EMBEDDING_MODEL_NAME || 'text-embedding-004',
    });

    // สร้าง Cache-backed embeddings เพื่อลดต้นทุนในการค้นหา
    const cacheStore = new InMemoryStore();
    const embeddings = CacheBackedEmbeddings.fromBytesStore(
      baseEmbeddings,
      cacheStore,
      {
        namespace: "search_embeddings" // กำหนด namespace แยกสำหรับการค้นหา
      }
    );

    const vectorStore = new SupabaseVectorStore(embeddings, {
        client: supabase,
        tableName: 'documents',
        queryName: 'match_documents'
    });

    // ===============================================
    // ค้นหาเอกสารที่คล้ายกัน
    // ===============================================
    const results = await vectorStore.similaritySearchWithScore(query, limit)
    
    console.log(`📋 พบผลลัพธ์: ${results.length} รายการ`)

    // ===============================================
    // จัดรูปแบบผลลัพธ์
    // ===============================================
    const formattedResults = results.map(([doc, score], index) => ({
      rank: index + 1,
      content: doc.pageContent,
      metadata: doc.metadata,
      relevance_score: score
    }))

    return NextResponse.json({
      query,
      results_count: results.length,
      results: formattedResults,
      success: true
    })

  } catch (error) {
    console.error('❌ Error ในการค้นหา:', error)
    
    return NextResponse.json({ 
      error: 'เกิดข้อผิดพลาดในการค้นหา',
      details: error instanceof Error ? error.message : 'Unknown error',
      success: false
    }, { status: 500 })
  }
}

/**
 * DELETE API: ลบข้อมูลทั้งหมดใน vector store
 */
export async function DELETE() {
  try {
    const supabase = await createClient();
    
    // ตรวจสอบจำนวนข้อมูลก่อนลบ
    const { count: existingCount } = await supabase
      .from('documents')
      .select('*', { count: 'exact', head: true });

    if (!existingCount || existingCount === 0) {
      return NextResponse.json({ 
        message: "ไม่พบข้อมูลในฐานข้อมูล - ไม่มีอะไรให้ลบ",
        deleted_records: 0,
        success: true
      })
    }

    console.log(`🗑️ กำลังลบข้อมูล ${existingCount} records...`);
    
    // ลบข้อมูลทั้งหมดในตาราง documents
    const { error } = await supabase
      .from('documents')
      .delete()
      .neq('id', 0) // ลบทุกแถวที่ id ไม่เท่ากับ 0 (ซึ่งคือทุกแถว)

    if (error) {
      throw new Error(error.message)
    }

    console.log(`✅ ลบข้อมูล ${existingCount} records สำเร็จ`)

    return NextResponse.json({ 
      message: `ลบข้อมูลใน vector store สำเร็จ - ลบไป ${existingCount} records`,
      deleted_records: existingCount,
      timestamp: new Date().toISOString(),
      success: true
    })

  } catch (error) {
    console.error('❌ Error ในการลบข้อมูล:', error)
    
    return NextResponse.json({ 
      error: 'เกิดข้อผิดพลาดในการลบข้อมูล',
      details: error instanceof Error ? error.message : 'Unknown error',
      success: false
    }, { status: 500 })
  }
}

/**
 * PUT API: ดูสถิติข้อมูลใน vector store
 */
export async function PUT() {
  try {
    const supabase = await createClient();
    
    // ตรวจสอบจำนวนข้อมูลทั้งหมด
    const { count: totalCount } = await supabase
      .from('documents')
      .select('*', { count: 'exact', head: true });

    if (!totalCount || totalCount === 0) {
      return NextResponse.json({ 
        message: "ไม่พบข้อมูลในฐานข้อมูล",
        stats: {
          total_records: 0,
          files_breakdown: [],
          timestamp: new Date().toISOString()
        },
        success: true
      })
    }

    // ดึงข้อมูล metadata เพื่อสร้างสถิติ
    const { data: documents } = await supabase
      .from('documents')
      .select('metadata')
      .limit(1000); // จำกัดไม่ให้เยอะเกินไป

    
    // กำหนด interface สำหรับ file stats
    interface FileStats {
      filename: string;
      type: string;
      chunks: number;
      total_chars: number;
    }

    const fileStats = documents?.reduce((acc: Record<string, FileStats>, doc) => {
      const filename = doc.metadata?.filename || 'unknown';
      const type = doc.metadata?.type || 'unknown';
      
      if (!acc[filename]) {
        acc[filename] = {
          filename,
          type,
          chunks: 0,
          total_chars: 0
        };
      }
      
      acc[filename].chunks += 1;
      acc[filename].total_chars += doc.metadata?.chunk_size || 0;
      
      return acc;
    }, {}) || {};

    const stats = {
      total_records: totalCount,
      files_breakdown: Object.values(fileStats),
      files_count: Object.keys(fileStats).length,
      timestamp: new Date().toISOString()
    };

    console.log(`📊 สถิติข้อมูล: ${totalCount} records จาก ${Object.keys(fileStats).length} ไฟล์`);

    return NextResponse.json({ 
      message: `พบข้อมูล ${totalCount} records จาก ${Object.keys(fileStats).length} ไฟล์`,
      stats,
      success: true
    })

  } catch (error) {
    console.error('❌ Error ในการดูสถิติข้อมูล:', error)
    
    return NextResponse.json({ 
      error: 'เกิดข้อผิดพลาดในการดูสถิติข้อมูล',
      details: error instanceof Error ? error.message : 'Unknown error',
      success: false
    }, { status: 500 })
  }
}