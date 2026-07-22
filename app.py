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
import gc  # Faxineiro de memória RAM do Python

# Configuração da página Web
st.set_page_config(page_title="Gerador Dm3 - Pro", page_icon="📦", layout="wide")

st.title("📦 Dm3 - Estúdio de Fotos Automático (Foco em Produtos)")
st.write("Suba a foto do material. O sistema isola, limpa itens encostados e centraliza automaticamente a mercadoria.")

# 1. Carregar o fundo padrão (Gabarito)
try:
    fundo_padrao = Image.open("fundo_dm3.png").convert("RGBA")
except FileNotFoundError:
    st.error("⚠️ Erro: O arquivo 'fundo_dm3.png' não foi encontrado na pasta do sistema!")
    st.stop()

# 2. Carregar Inteligência Artificial (Dinâmico: Geral ou Especialista em Roupas)
@st.cache_resource
def carregar_modelo_ia(modo_roupa=False):
    try:
        if modo_roupa:
            # IA especialista em moda: isola tecidos e ignora pescoço/corpo de manequins
            return new_session("u2net_cloth_seg")
        else:
            # IA geral: excelente para caixas, totens, ferramentas e produtos em geral
            return new_session("isnet-general-use")
    except Exception:
        return new_session("u2netp")

# 3. FUNÇÃO INTELIGENTE: Remove itens encostados usando Núcleo Seguro
def limpar_itens_encostados(img_rgba, forca_corte):
    img_array = np.array(img_rgba)
    canal_alpha = img_array[:, :, 3]
    
    # Cria uma máscara binária pura
    _, binaria = cv2.threshold(canal_alpha, 20, 255, cv2.THRESH_BINARY)
    
    if forca_corte > 0:
        k_size = int(forca_corte)
        if k_size % 2 == 0:
            k_size += 1
        elemento_estrutura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        # Erosão temporária para quebrar pontes finas (arames de clipes, post-its)
        mascara_erodida = cv2.erode(binaria, elemento_estrutura, iterations=1)
        
        # Achar os blocos separados e manter APENAS o corpo principal
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara_erodida, connectivity=8)
        if num_labels > 1:
            maior_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            nucleo_produto = np.where(labels == maior_label, 255, 0).astype(np.uint8)
        else:
            nucleo_produto = mascara_erodida
            
        # Dilata o núcleo de volta para devolver a borda original e lisa do produto
        mascara_restaurada = cv2.dilate(nucleo_produto, elemento_estrutura, iterations=1)
        alpha_final = cv2.bitwise_and(canal_alpha, mascara_restaurada)
    else:
        alpha_final = canal_alpha
        
    alpha_suave = cv2.GaussianBlur(alpha_final, (3, 3), 0)
    img_array[:, :, 3] = alpha_suave
    
    return Image.fromarray(img_array)

# 4. Função de Recorte com TRAVA DE MEMÓRIA (max_entries=2 impede o servidor de cair!)
@st.cache_data(max_entries=2, show_spinner=False)
def recortar_fundo(imagem_bytes, usar_alta_precisao, forca_desconexao, modo_roupa):
    img = Image.open(io.BytesIO(imagem_bytes)).convert("RGBA")
    sessao_ia = carregar_modelo_ia(modo_roupa)
    
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
        
    recorte_limpo = limpar_itens_encostados(recorte_bruto, forca_desconexao)
    return recorte_limpo

# --- BARRA LATERAL DE CONTROLES ---
st.sidebar.header("🚀 Desempenho e RAM")
if st.sidebar.button("🧹 Limpar Memória RAM", use_container_width=True, help="Clique aqui se o sistema estiver rodando há muito tempo e parecer um pouco lento."):
    st.cache_data.clear()
    gc.collect()
    st.sidebar.success("Memória do servidor limpa!")

st.sidebar.markdown("---")
st.sidebar.header("👕 Tipo de Produto")
modo_roupa = st.sidebar.checkbox(
    "Modo Roupa / Extrair do Manequim", 
    value=False, 
    help="Marque ESTA CAIXA ao fotografar roupas em manequins, cabides ou modelos. O sistema usará uma IA especialista em tecidos para apagar pescoços, braços e suportes!"
)

st.sidebar.markdown("---")
st.sidebar.header("🛠️ Ajustes de Limpeza")
forca_sep = st.sidebar.slider(
    "✂️ Força para Desgrudar Clipes e Papéis", 
    min_value=0, 
    max_value=40, 
    value=0, 
    step=2, 
    help="Deixe em 0 se a foto não tiver nada encostado. Se houver clipes ou papéis colados no produto, aumente gradualmente até eles sumirem."
)
modo_precisao = st.sidebar.checkbox("✨ Modo Alta Precisão (Bordas mais suaves)", value=False)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Modo de Centralização")
modo_centro_massa = st.sidebar.checkbox(
    "⚖️ Alinhar por Centro de Gravidade", 
    value=False, 
    help="Ative para objetos assimétricos ou inclinados. O sistema calcula o peso visual para a imagem não parecer 'pendendo' para um lado."
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
        # Passando o parâmetro do modo roupa para o recorte
        img_sem_fundo = recortar_fundo(bytes_arquivo, modo_precisao, forca_sep, modo_roupa)
        
        # --- CORTE ÓPTICO ANTI-FANTASMA (Ignora sombras invisíveis que desalinham a foto) ---
        img_arr = np.array(img_sem_fundo)
        alpha_canal = img_arr[:, :, 3]
        y_indices, x_indices = np.where(alpha_canal > 35) # Só corta o que tem opacidade real
        
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
        
        # --- CÁLCULO DE CENTRALIZAÇÃO AUTOMÁTICA ---
        if modo_centro_massa:
            # Calcula o Centro de Massa (Gravidade) dos pixels visíveis
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
            # Centralização Geométrica Limpa (Padrão de Estúdio)
            posicao_x = ((largura_fundo - nova_largura) // 2) + ajuste_x
            posicao_y = ((altura_fundo - nova_altura) // 2) + ajuste_y
        
        imagem_final = fundo_padrao.copy()
        imagem_final.paste(img_redimensionada, (posicao_x, posicao_y), img_redimensionada)
        
        buf = io.BytesIO()
        imagem_final.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        # Faxineiro da RAM trabalhando após gerar a foto
        gc.collect()

    st.success("✅ Imagem processada e centralizada com sucesso!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(arquivo_enviado, caption="Foto Original", use_container_width=True)
    with col2:
        st.image(imagem_final, caption="Resultado Final Dm3 (Centralização Automática)", use_container_width=True)
        
    st.download_button(
        label="⬇️ Baixar Imagem Pronta",
        data=byte_im,
        file_name="produto_dm3_pronto.png",
        mime="image/png",
        use_container_width=True
    )
