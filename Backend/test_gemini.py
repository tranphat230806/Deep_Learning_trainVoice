import google.generativeai as genai

# Thay API Key của bạn vào đây
genai.configure(api_key="AIzaSyDHtbwJNMOrnYejcPCGGkxLXbwwCsdxyGQ")

print("🔍 Đang quét các mô hình AI có sẵn trong API của bạn...\n")

# Vòng lặp in ra toàn bộ tên model có hỗ trợ tạo văn bản
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"✅ Tên dùng cho Code: {m.name}")