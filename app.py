import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# --- Yapılandırma ve Modellerin Yüklenmesi ---

# 1. Metin Tespit Modeli (LLM Detection)
# Hugging Face'den fine-tune edilmiş bir RoBERTa modeli kullanacağız.
# Bu model, GPT-2/3 gibi modeller tarafından üretilen metinleri sınıflandırmak için eğitilmiştir.
TEXT_MODEL_NAME = "roberta-base-openai-detector" # Alternatif: "Hello-SimpleAI/chatgpt-detector-roberta"
print(f"Metin modeli yükleniyor: {TEXT_MODEL_NAME}...")
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME)
print("Metin modeli yüklendi.")

# 2. Görüntü Tespit Modeli (AI/GAN Detection)
# Bu örnek için, önceden eğitilmiş bir ResNet-18 modelini basitleştirilmiş bir sınıflandırıcısı olarak kullanacağız.
# Gerçek dünyada bu, GAN veya Diffusion modelleri üzerinde eğitilmiş özel bir model olmalıdır.
print("Görüntü modeli yükleniyor: ResNet-18 tabanlı...")
image_model = models.resnet18(pretrained=True)
num_ftrs = image_model.fc.in_features
image_model.fc = torch.nn.Linear(num_ftrs, 2) # İkili sınıflandırma (İnsan / AI)
# Not: Bu modelin gerçek tespiti için kendi verilerinizle fine-tune edilmesi gerekir.
# Şimdilik, modelin mimarisini ve çıktı mantığını simüle ediyoruz.
image_model.eval()
print("Görüntü modeli yüklendi.")

# Görüntü ön işleme için dönüşümler
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Analiz Fonksiyonları ---

def detect_text_source(text):
    """Metnin AI tarafından mı yazıldığını tespit eder."""
    if not text or len(text.strip()) == 0:
        return "Lütfen metin girin."
    
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = text_model(**inputs)
    
    # logits'i olasılıklara dönüştür (softmax)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Sınıf etiketleri modele bağlıdır. RoBERTa modeli için: [Human, AI]
    human_prob = probs[0][0].item()
    ai_prob = probs[0][1].item()
    
    print(f"Metin Analizi: İnsan %{human_prob*100:.2f}, AI %{ai_prob*100:.2f}")
    
    if ai_prob > 0.5:
        return f"🕵️‍♂️ TESPİT EDİLDİ: Metin muhtemelen YAPAY ZEKA tarafından yazılmış. (%{ai_prob*100:.1f})"
    else:
        return f"👨‍💻 TESPİT EDİLDİ: Metin muhtemelen İNSAN tarafından yazılmış. (%{human_prob*100:.1f})"

def detect_image_source(image):
    """Görüntünün AI tarafından mı oluşturulduğunu tespit eder."""
    if image is None:
        return "Lütfen bir görüntü yükleyin."
    
    # Görüntüyü hazırla ve tahmin et
    image_tensor = image_transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = image_model(image_tensor)
    
    # Sınıf etiketleri: [Human, AI] (Fine-tune edilmiş bir modelde bu etiketler nettir)
    # Şimdilik, modelin çıktısını olasılıklara dönüştürüyoruz.
    probs = torch.nn.functional.softmax(outputs, dim=-1)
    human_prob = probs[0][0].item()
    ai_prob = probs[0][1].item()
    
    print(f"Görüntü Analizi: İnsan %{human_prob*100:.2f}, AI %{ai_prob*100:.2f}")
    
    # Not: Önceden eğitilmiş ResNet, AI tespiti için eğitilmediği için buradaki sonuçlar
    # tutarlı olmayabilir. Bu sadece prototipin nasıl çalışacağını gösteren bir örnektir.
    if ai_prob > 0.6: # Eşik değerini prototip için yükselttik
        return f"🕵️‍♂️ TESPİT EDİLDİ: Görüntü muhtemelen YAPAY ZEKA tarafından oluşturulmuş. (%{ai_prob*100:.1f})"
    else:
        return f"📷 TESPİT EDİLDİ: Görüntü muhtemelen GERÇEK (İNSAN) yapımı. (%{human_prob*100:.1f})"

# --- Gradio Arayüzü Tasarımı ---

with gr.Blocks(theme=gr.themes.Soft(), title="ContentSourceDetector") as demo:
    gr.Markdown("# Multimodal İçerik Kaynağı Tespit Uygulaması")
    gr.Markdown("İçeriğin (metin veya görüntü) yapay zeka tarafından mı yoksa gerçek bir insan tarafından mı oluşturulduğunu analiz edin.")
    
    with gr.Tab("Metin Analizi (GPT-4/LLM Detector)"):
        with gr.Row():
            text_input = gr.Textbox(lines=5, label="Analiz edilecek Metni Girin", placeholder="Örn: Yapay zekanın geleceği hakkında bir metin...")
            text_output = gr.Textbox(label="Metin Kaynağı Sonucu")
        text_button = gr.Button("Metni Analiz Et", variant="primary")
        text_button.click(detect_text_source, inputs=text_input, outputs=text_output)
    
    with gr.Tab("Görüntü Analizi (Deepfake/GAN Detector)"):
        with gr.Row():
            image_input = gr.Image(label="Analiz edilecek Görüntüyü Yükleyin", type="pil")
            image_output = gr.Textbox(label="Görüntü Kaynağı Sonucu")
        image_button = gr.Button("Görüntüyü Analiz Et", variant="primary")
        image_button.click(detect_image_source, inputs=image_input, outputs=image_output)
        
    gr.Markdown("""
    ### Prototip Notları ve Sınırlamalar:
    * **Metin Tespit:** OpenAI'ın detektör modeline dayanan RoBERTa tabanlı bir model kullanır. Genellikle iyi çalışır ancak güncel GPT-4 gibi modellerde doğruluğu azalabilir.
    * **Görüntü Tespit:** Bu prototip, standart bir ResNet-18 modelini kullanır. **Gerçek bir AI üretimi tespiti için bu modelin özel verilerle (GAN/Diffusion) fine-tune edilmesi zorunludur.** Buradaki sonuçlar sadece API yapısının gösterimidir.
    * Sistem, Google Colab'da `cuda` (GPU) mevcutsa otomatik olarak hızlanacaktır.
    """
)

# --- Uygulamayı Çalıştır ---
if __name__ == "__main__":
    # demo.launch(share=True) # Public URL oluşturmak için 'share=True' ekleyin
    demo.launch()