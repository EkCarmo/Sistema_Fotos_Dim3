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
st.write("Suba a foto do material. O sistema isola, limpa fundos complexos e centraliza automaticamente a mercadoria.")

# 1. Carregar o fundo padrão (Gabarito)
try:
    fundo_padrao = Image.open("fundo_dm3.png").convert("RGBA")
except FileNotFoundError:
    st.error("⚠️ Erro: O arquivo 'fundo_dm3.png' não foi encontrado na pasta do sistema!")
    st.stop()

# 2. Carregar Inteligência Artificial (Estável para nuvem)
@st.cache_resource
def carregar_modelo_ia():
    try:
        return new_session("isnet-general-use")
    except Exception:
        return new_session("u2netp")

# 3. FUNÇÃO INTELIGENTE: Limpa névoas, móveis soltos e respeita vidros/kits
def limpar_itens_encostados(img_rgba, forca_corte, modo_transparente, filtro_tamanho_pct):
    img_array = np.array(img_rgba)
    canal_alpha = img_array[:, :, 3]
    
    # 1. TRAVA DE VIDRO / TRANSPARÊNCIA:
    # Se for vidro/acrílico, usamos um corte muito suave (10) para não apagar a transparência do material.
    # Se for produto comum, usamos o corte anti-névoa (180) para apagar a fumaça cinza do galpão.
    limite_alpha = 10 if modo_transparente else 180
    _, binaria = cv2.threshold(canal_alpha, limite_alpha, 255, cv2.THRESH_BINARY)
    
    # 2. SEPARAÇÃO E CORROSÃO (Para soltar conexões com móveis ou clipes)
    if forca_corte > 0:
        k_size = int(forca_corte)
        if k_size % 2 == 0:
            k_size += 1
        elemento_estrutura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        mascara_trabalho = cv2.erode(binaria, elemento_estrutura, iterations=1)
    else:
        elemento_estrutura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mascara_trabalho = cv2.erode(binaria, elemento_estrutura, iterations=1)
        
    # 3. FILTRO DE ILHAS E KITS (Controlado pelo slider de porcentagem):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara_trabalho, connectivity=8)
    if num_labels > 1:
        area_maxima = np.max(stats[1:, cv2.CC_STAT_AREA])
        
        # Se o slider estiver em 0%, mantém todas as peças soltas (ideal para kits com parafusos/controles soltos).
        # Se estiver maior que 0%, apaga caixas, banquetas e sujeiras menores que a porcentagem escolhida.
        if filtro_tamanho_pct > 0:
            fator = filtro_tamanho_pct / 100.0
            areas_validas = np.where(stats[1:, cv2.CC_STAT_AREA] >= (area_maxima * fator))[0] + 1
            nucleo_produto = np.isin(labels, areas_validas).astype(np.uint8) * 255
        else:
            nucleo_produto = mascara_trabalho
    else:
        nucleo_produto = mascara_trabalho
        
    # 4. DILATAÇÃO DE RESTAURAÇÃO (Devolve a borda original e lisa do produto)
    mascara_restaurada = cv2.dilate(nucleo_produto, elemento_estrutura, iterations=1)
    
    # Cruza a máscara limpa com o alpha original para manter o acabamento profissional
    alpha_final = cv2.bitwise_and(canal_alpha, mascara_restaurada)
    
    # Suavização leve de acabamento anti-serrilhado
    alpha_suave = cv2.GaussianBlur(alpha_final, (3, 3), 0)
    img_array[:, :, 3] = alpha_suave
    
    return Image.fromarray(img_array)

# 4. Função de Recorte Combinada
@st.cache_data(show_spinner=False)
def recortar_fundo(imagem_bytes, usar_alta_precisao, forca_desconexao, modo_transp, filtro_tam):
    img = Image.open(io.BytesIO(imagem_bytes)).convert("RGBA")
    sessao_ia = carregar_modelo_ia()
    
    if usar_alta_precisao:
        recorte_bruto = remove(
            img, 
            session=sessao_ia, 
            alpha_matting=True, 
            alpha_matting_foreground_threshold=250, 
            alpha_matting_background_threshold=10
        )
    else:
        recorte_bruto = remove(img, session=sessao_ia)
        
    recorte_limpo = limpar_itens_encostados(recorte_bruto, forca_desconexao, modo_transp, filtro_tam)
    return recorte_limpo

# --- BARRA LATERAL DE CONTROLES ---
st.sidebar.header("🛡️ Proteções Especiais")
modo_vidro = st.sidebar.checkbox(
    "🍾 Produto de Vidro / Transparente", 
    value=False, 
    help="Marque ESTA CAIXA se o produto for de vidro, acrílico ou tiver partes transparentes. Isso impede que o sistema apague a transparência natural do material."
)

filtro_ilhas = st.sidebar.slider(
    "🧹 Limpeza de Objetos Soltos ao Fundo (%)", 
    min_value=0, 
    max_value=30, 
    value=10, 
    step=2, 
    help="Apaga objetos desconectados menores que essa porcentagem (ex: madeiras, caixas ao fundo). ATENÇÃO: Deixe em 0% se estiver fotografando um KIT com peças pequenas soltas na mesa!"
)

st.sidebar.markdown("---")
st.sidebar.header("🛠️ Ajustes de Limpeza")
forca_sep = st.sidebar.slider(
    "✂️ Força para Desgrudar Clipes e Móveis", 
    min_value=0, 
    max_value=30, 
    value=0, 
    step=2, 
    help="Deixe em 0 para displays e caixas limpas. Se algum móvel ou clipe grudado insistir em aparecer, aumente gradualmente (ex: 6, 10 ou 14)."
)
modo_precisao = st.sidebar.checkbox("✨ Modo Alta Precisão (Bordas mais suaves)", value=False)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Modo de Centralização")
modo_centro_massa = st.sidebar.checkbox(
    "⚖️ Alinhar por Centro de Gravidade", 
    value=False, 
    help="Ative para objetos assimétricos ou inclinados."
)

st.sidebar.markdown("---")
st.sidebar.header("📐 Ajuste de Posição e Tamanho")
escala_manual = st.sidebar.slider("Tamanho do Produto (%)", min_value=30, max_value=100, value=80, step=5)
ajuste_x = st.sidebar.slider("Mover para Horizontal (↔)", min_value=-200, max_value=200, value=0, step=10)
ajuste_y = st.sidebar.slider("Mover para Vertical (↕)", min_value=-200, max_value=200, value=0, step=10)

# --- ÁREA PRINCIPAL ---
arquivo_enviado = st.file_uploader("Selecione ou arraste a foto do produto aqui:", type=["png", "jpg", "jpeg"])

if arquivo_enviado is not None:
    bytes_arquivo = arquivo_enviado.getvalue()
    
    with st.spinner("🤖 IA processando, isolando e centralizando o produto..."):
        img_sem_fundo = recortar_fundo(bytes_arquivo, modo_precisao, forca_sep, modo_vidro, filtro_ilhas)
        
        # Corte Óptico Anti-Fantasma (Respeita vidros se a opção estiver ativa)
        img_arr = np.array(img_sem_fundo)
        alpha_canal = img_arr[:, :, 3]
        corte_alpha_min = 10 if modo_vidro else 50
        y_indices, x_indices = np.where(alpha_canal > corte_alpha_min)
        
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            img_cortada = img_sem_fundo.crop((x_min, y_min, x_max + 1, y_max + 1))
        else:
            img_cortada = img_sem_fundo

        largura_fundo, altura_fundo = fundo_padrao.size
        fator_escala = escala_manual / 100.0
        
        proporcao = min((largura_fundo * fator_escala) / img_cortada.width, (altura_fundo * fator_escala) / img_cortada.height)
        nova_largura = int(img_cortada.width * proporcao)
        nova_altura = int(img_cortada.height * proporcao)
        
        img_redimensionada = img_cortada.resize((nova_largura, nova_altura), Image.Resampling.LANCZOS)
        
        if modo_centro_massa:
            arr_redim = np.array(img_redimensionada)[:, :, 3]
            momentos = cv2.moments(arr_redim)
            if momentos["m00"] != 0:
                centro_x = int(momentos["m10"] / momentos["m00"])
                centro_y = int(momentos["m01"] / momentos["m00"])
                posicao_x = (largura_fundo // 2) - centro_x + ajuste_x
                posicao_y = (altura_fundo // 2) - centro_y + ajuste_y
            else:
                posicao_x = ((largura_fundo - nova_largura) // 2) + ajuste_x
                posicao_y = ((altura_fundo - nova_altura) // 2) + ajuste_y
        else:
            posicao_x = ((largura_fundo - nova_largura) // 2) + ajuste_x
            posicao_y = ((altura_fundo - nova_altura) // 2) + ajuste_y
        
        imagem_final = fundo_padrao.copy()
        imagem_final.paste(img_redimensionada, (posicao_x, posicao_y), img_redimensionada)
        
        buf = io.BytesIO()
        imagem_final.save(buf, format="PNG")
        byte_im = buf.getvalue()

    st.success("✅ Imagem processada, limpa e centralizada com sucesso!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(arquivo_enviado, caption="Foto Original", use_container_width=True)
    with col2:
        st.image(imagem_final, caption="Resultado Final Dm3 (Fundo e Névoas Eliminados)", use_container_width=True)
        
    st.download_button(
        label="⬇️ Baixar Imagem Pronta",
        data=byte_im,
        file_name="produto_dm3_pronto.png",
        mime="image/png",
        use_container_width=True
    )
