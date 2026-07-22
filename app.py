import os
# --- TRAVAS ANTI-CONGELAMENTO PARA NUVEM GRATUITA ---
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
st.write("Suba a foto do material. O sistema remove o fundo e elimina sombras e itens encostados sem travar.")

# 1. Carregar o fundo padrão (Gabarito)
try:
    fundo_padrao = Image.open("fundo_dm3.png").convert("RGBA")
except FileNotFoundError:
    st.error("⚠️ Erro: O arquivo 'fundo_dm3.png' não foi encontrado na pasta do sistema!")
    st.stop()

# 2. Carregar Inteligência Artificial (Modelo leve e estável)
@st.cache_resource
def carregar_modelo_ia():
    try:
        return new_session("isnet-general-use")
    except Exception:
        return new_session("u2netp")

# 3. FUNÇÃO INTELIGENTE: Limpa sombras fantasmas e apaga itens claros (clipes/papéis) encostados
def limpar_sombras_e_itens(img_rgba, forca_limpeza):
    img_array = np.array(img_rgba)
    r, g, b, alpha = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2], img_array[:, :, 3]
    
    # 1. CORTE DE SOMBRAS (Anti-Fantasma):
    # Elimina transparências fracas (como a sombra branca do post-it), deixando a borda 100% nítida e sólida
    _, alpha_limpo = cv2.threshold(alpha, 200, 255, cv2.THRESH_BINARY)
    
    # 2. LIMPEZA DE ITENS ENCOSTADOS (Ativada pelo Slider):
    # Se o slider for maior que 0, ele identifica objetos claros/metálicos (clipes) grudados no produto e os apaga
    if forca_limpeza > 0:
        # Transforma para escala de cinza para medir o brilho dos objetos
        cinza = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2GRAY)
        
        # Cria uma máscara apagando tudo o que for mais claro que o nível escolhido no slider
        limite_brilho = 255 - int(forca_limpeza * 2)
        _, mascara_sem_clipes = cv2.threshold(cinza, limite_brilho, 255, cv2.THRESH_BINARY_INV)
        
        # Combina a limpeza de borda com a remoção de clipes
        alpha_limpo = cv2.bitwise_and(alpha_limpo, mascara_sem_clipes)
        
        # Pega apenas o maior objeto restante (o produto real), jogando fora pedaços soltos
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_limpo, connectivity=8)
        if num_labels > 1:
            maior_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            alpha_limpo = np.where(labels == maior_label, 255, 0).astype(np.uint8)
            
    # Suavização mínima de meio pixel apenas para a borda não ficar pontiaguda
    alpha_final = cv2.GaussianBlur(alpha_limpo, (3, 3), 0)
    img_array[:, :, 3] = alpha_final
    
    return Image.fromarray(img_array)

# 4. Função de Recorte Combinada
@st.cache_data(show_spinner=False)
def recortar_fundo(imagem_bytes, usar_alta_precisao, forca_limpeza):
    img = Image.open(io.BytesIO(imagem_bytes)).convert("RGBA")
    sessao_ia = carregar_modelo_ia()
    
    # Rodamos a IA limpa
    recorte_bruto = remove(img, session=sessao_ia)
        
    # Passamos no nosso filtro leve de remoção de sombras e clipes
    recorte_limpo = limpar_sombras_e_itens(recorte_bruto, forca_limpeza)
    return recorte_limpo

# --- BARRA LATERAL DE CONTROLES ---
st.sidebar.header("🛠️ Ajustes de Limpeza")
forca_sep = st.sidebar.slider(
    "✂️ Força para Apagar Clipes/Papéis", 
    min_value=0, 
    max_value=50, 
    value=0, 
    step=5, 
    help="Deixe em 0 para limpar apenas sombras. Aumente se quiser apagar clipes metálicos ou papéis claros encostados no produto."
)
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
    
    with st.spinner("🤖 IA processando e eliminando sombras fantasmas..."):
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

    st.success("✅ Imagem processada com bordas nítidas e sem sombras!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(arquivo_enviado, caption="Foto Original", use_container_width=True)
    with col2:
        st.image(imagem_final, caption="Resultado Final Dm3 (Sem Sombras Fantasmas)", use_container_width=True)
        
    st.download_button(
        label="⬇️ Baixar Imagem Pronta",
        data=byte_im,
        file_name="produto_dm3_pronto.png",
        mime="image/png",
        use_container_width=True
    )
