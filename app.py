import os
# --- TRAVA DE SEGURANÇA PARA NUVEM ---
# Garante que o processador virtual da nuvem não engasgue ao abrir a IA
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
from rembg import remove, new_session
from PIL import Image
import io

# Configuração da página
st.set_page_config(page_title="Gerador Dm3 - Pro", page_icon="📦", layout="wide")

st.title("📦 Dm3 - Estúdio de Fotos Automático")
st.write("Suba a foto do produto. O modelo especialista isola a mercadoria e ignora o fundo do galpão.")

# 1. Carregar o fundo padrão (Gabarito)
try:
    fundo_padrao = Image.open("fundo_dm3.png").convert("RGBA")
except FileNotFoundError:
    st.error("⚠️ Erro: O arquivo 'fundo_dm3.png' não foi encontrado na pasta do sistema!")
    st.stop()

# 2. CARREGAR A IA ESTÁVEL (Especialista em Produtos / Estoque)
@st.cache_resource
def carregar_modelo_ia():
    try:
        # O modelo que resolveu o problema do post-it e não traça o servidor
        return new_session("isnet-general-use")
    except Exception:
        # Fallback de segurança ultra-leve
        return new_session("u2netp")

# 3. FUNÇÃO DE RECORTE LIMPO
@st.cache_data(show_spinner=False)
def recortar_fundo(imagem_bytes, usar_alta_precisao):
    img = Image.open(io.BytesIO(imagem_bytes)).convert("RGBA")
    sessao_ia = carregar_modelo_ia()
    
    if usar_alta_precisao:
        return remove(
            img, 
            session=sessao_ia, 
            alpha_matting=True, 
            alpha_matting_foreground_threshold=240, 
            alpha_matting_background_threshold=10
        )
    else:
        return remove(img, session=sessao_ia)

# --- BARRA LATERAL DE CONTROLES ---
st.sidebar.header("🛠️ Ajustes do Recorte")
modo_precisao = st.sidebar.checkbox("✨ Modo Alta Precisão (Bordas mais suaves)", value=False)

st.sidebar.markdown("---")
st.sidebar.header("📐 Ajuste de Posição e Tamanho")
escala_manual = st.sidebar.slider("Tamanho do Produto (%)", min_value=30, max_value=100, value=80, step=5)
ajuste_x = st.sidebar.slider("Mover para Horizontal (↔)", min_value=-200, max_value=200, value=0, step=10)
ajuste_y = st.sidebar.slider("Mover para Vertical (↕)", min_value=-200, max_value=200, value=0, step=10)

# --- ÁREA PRINCIPAL ---
arquivo_enviado = st.file_uploader("Selecione ou arraste a foto do produto aqui:", type=["png", "jpg", "jpeg"])

if arquivo_enviado is not None:
    bytes_arquivo = arquivo_enviado.getvalue()
    
    with st.spinner("🤖 IA isolando o produto e removendo o fundo..."):
        img_sem_fundo = recortar_fundo(bytes_arquivo, modo_precisao)
        
        # Auto-crop para tirar espaços vazios transparentes ao redor
        caixa_delimitadora = img_sem_fundo.getbbox()
        if caixa_delimitadora:
            img_cortada = img_sem_fundo.crop(caixa_delimitadora)
        else:
            img_cortada = img_sem_fundo

        largura_fundo, altura_fundo = fundo_padrao.size
        fator_escala = escala_manual / 100.0
        
        proporcao = min((largura_fundo * fator_escala) / img_cortada.width, (altura_fundo * fator_escala) / img_cortada.height)
        nova_largura = int(img_cortada.width * proporcao)
        nova_altura = int(img_cortada.height * proporcao)
        
        img_redimensionada = img_cortada.resize((nova_largura, nova_altura), Image.Resampling.LANCZOS)
        
        posicao_x = ((largura_fundo - nova_largura) // 2) + ajuste_x
        posicao_y = ((altura_fundo - nova_altura) // 2) + ajuste_y
        
        imagem_final = fundo_padrao.copy()
        imagem_final.paste(img_redimensionada, (posicao_x, posicao_y), img_redimensionada)
        
        buf = io.BytesIO()
        imagem_final.save(buf, format="PNG")
        byte_im = buf.getvalue()

    st.success("✅ Imagem gerada com sucesso!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(arquivo_enviado, caption="Foto Original", use_container_width=True)
    with col2:
        st.image(imagem_final, caption="Resultado Final Dm3", use_container_width=True)
        
    st.download_button(
        label="⬇️ Baixar Imagem Pronta",
        data=byte_im,
        file_name="produto_dm3_pronto.png",
        mime="image/png",
        use_container_width=True
    )
