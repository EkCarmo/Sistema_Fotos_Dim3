import os
# --- TRAVAS ANTI-CONGELAMENTO PARA NUVEM GRATUITA ---
# Deve vir ANTES de importar o rembg/cv2 para evitar que a CPU virtual engasgue
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
from rembg import remove, new_session
from PIL import Image
import io
import cv2
import numpy as np

# Configuração da página Web
st.set_page_config(page_title="Gerador Dm3 - Pro", page_icon="📦", layout="wide")

st.title("📦 Dm3 - Estúdio de Fotos Automático (Foco em Produtos)")
st.write("Suba a foto do material. O sistema remove o fundo mantendo bordas limpas e nítidas.")

# 1. Carregar o fundo padrão (Gabarito)
try:
    fundo_padrao = Image.open("fundo_dm3.png").convert("RGBA")
except FileNotFoundError:
    st.error("⚠️ Erro: O arquivo 'fundo_dm3.png' não foi encontrado na pasta do sistema!")
    st.stop()

# 2. Carregar IA (DIRETO NO MODELO LEVE - 40 MB - Impossível travar por memória!)
@st.cache_resource
def carregar_modelo_ia():
    # O modelo 'u2netp' é o mais leve e rápido da categoria, perfeito para nuvem gratuita
    return new_session("u2netp")

# 3. FUNÇÃO DE LIMPEZA SUAVE: Isola o produto sem mastigar a borracha
def limpar_bordas_e_isolamento(img_rgba, forca_corte):
    img_array = np.array(img_rgba)
    canal_alpha = img_array[:, :, 3]
    
    # Cria máscara preta e branca pura
    _, binaria = cv2.threshold(canal_alpha, 10, 255, cv2.THRESH_BINARY)
    
    # 1. Aplica fechamento morfológico leve APENAS se o usuário pedir força no slider
    if forca_corte > 0:
        k_size = int(forca_corte)
        if k_size % 2 == 0:
            k_size += 1
        tamanho_corte = np.ones((k_size, k_size), np.uint8)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, tamanho_corte)
    
    # 2. Encontra os contornos na tela para isolar o objeto principal
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return img_rgba
        
    # Identifica o contorno de maior área (a mercadoria real)
    maior_contorno = max(contornos, key=cv2.contourArea)
    
    # Desenha uma máscara sólida contendo apenas o produto principal
    mascara_limpa = np.zeros_like(binaria)
    cv2.drawContours(mascara_limpa, [maior_contorno], -1, 255, thickness=cv2.FILLED)
    
    # 3. Anti-serrilhado: Suaviza a borda final para não parecer recorte mal feito
    mascara_suavizada = cv2.GaussianBlur(mascara_limpa, (5, 5), 0)
    
    # Aplica a máscara limpa na imagem original
    img_array[:, :, 3] = cv2.bitwise_and(canal_alpha, canal_alpha, mask=mascara_suavizada)
    
    return Image.fromarray(img_array)

# 4. Função de Recorte Combinada
@st.cache_data(show_spinner=False)
def recortar_fundo(imagem_bytes, usar_alta_precisao, forca_desconexao):
    img = Image.open(io.BytesIO(imagem_bytes)).convert("RGBA")
    sessao_ia = carregar_modelo_ia()
    
    if usar_alta_precisao:
        recorte_bruto = remove(
            img, 
            session=sessao_ia, 
            alpha_matting=True, 
            alpha_matting_foreground_threshold=240, 
            alpha_matting_background_threshold=10
        )
    else:
        recorte_bruto = remove(img, session=sessao_ia)
        
    # Limpa contornos e elimina clipes/sujeiras isoladas
    recorte_limpo = limpar_bordas_e_isolamento(recorte_bruto, forca_desconexao)
    return recorte_limpo

# --- BARRA LATERAL DE CONTROLES ---
st.sidebar.header("🛠️ Ajustes de Limpeza")
forca_sep = st.sidebar.slider(
    "✂️ Força para Desgrudar Itens Finos", 
    min_value=0, 
    max_value=30, 
    value=0, 
    step=3, 
    help="Deixe em 0 para a borda ficar perfeitamente lisa. Aumente apenas se clipes ou arames teimosos não sumirem sozinhos."
)
modo_precisao = st.sidebar.checkbox("✨ Modo Alta Precisão (Bordas mais suaves)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("📐 Ajuste de Posição e Tamanho")
escala_manual = st.sidebar.slider("Tamanho do Produto (%)", min_value=30, max_value=100, value=80, step=5)
ajuste_x = st.sidebar.slider("Mover para Horizontal (↔)", min_value=-200, max_value=200, value=0, step=10)
ajuste_y = st.sidebar.slider("Mover para Vertical (↕)", min_value=-200, max_value=200, value=0, step=10)

# --- ÁREA PRINCIPAL ---
arquivo_enviado = st.file_uploader("Selecione ou arraste a foto do produto aqui:", type=["png", "jpg", "jpeg"])

if arquivo_enviado is not None:
    bytes_arquivo = arquivo_enviado.getvalue()
    
    with st.spinner("🤖 IA processando contornos de alta precisão..."):
        img_sem_fundo = recortar_fundo(bytes_arquivo, modo_precisao, forca_sep)
        
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

    st.success("✅ Imagem processada com contornos limpos!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(arquivo_enviado, caption="Foto Original", use_container_width=True)
    with col2:
        st.image(imagem_final, caption="Resultado Final Dm3 (Alta Nitidez)", use_container_width=True)
        
    st.download_button(
        label="⬇️ Baixar Imagem Pronta",
        data=byte_im,
        file_name="produto_dm3_pronto.png",
        mime="image/png",
        use_container_width=True
    )
