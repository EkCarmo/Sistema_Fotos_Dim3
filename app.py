import streamlit as st
from rembg import remove, new_session
from PIL import Image
import io
import cv2
import numpy as np

# Configuração da página Web
st.set_page_config(page_title="Gerador Dm3 - Pro", page_icon="📦", layout="wide")

st.title("📦 Dm3 - Estúdio de Fotos Automático (Foco em Produtos)")
st.write("Suba a foto do material. O sistema remove o fundo e elimina itens encostados ou soltos na cena.")

# 1. Carregar o fundo padrão (Gabarito)
try:
    fundo_padrao = Image.open("fundo_dm3.png").convert("RGBA")
except FileNotFoundError:
    st.error("⚠️ Erro: O arquivo 'fundo_dm3.png' não foi encontrado na pasta do sistema!")
    st.stop()

# 2. Carregar Inteligência Artificial (Modelo Avançado)
@st.cache_resource
def carregar_modelo_ia():
    try:
        return new_session("birefnet-general-lite")
    except Exception:
        return new_session("isnet-general-use")

# 3. FUNÇÃO MATEMÁTICA: Arrebenta conexões com força dinâmica programável
def limpar_sujeiras_e_itens_encostados(img_rgba, forca_corte):
    # Se a força for 0, não aplica o filtro
    if forca_corte <= 0:
        return img_rgba
        
    img_array = np.array(img_rgba)
    canal_alpha = img_array[:, :, 3]
    
    # Cria máscara preta e branca pura
    _, binaria = cv2.threshold(canal_alpha, 10, 255, cv2.THRESH_BINARY)
    
    # Garante que o tamanho da "tesoura" matemática seja um número ímpar
    k_size = int(forca_corte)
    if k_size % 2 == 0:
        k_size += 1
    tamanho_corte = np.ones((k_size, k_size), np.uint8)
    
    # PASSO A: Abertura Morfológica (Erosão seguida de Dilatação)
    # Isso destrói arames de clipes, post-its e pontes finas sem encolher o produto principal
    mask_aberta = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, tamanho_corte)
    
    # PASSO B: Encontrar todos os pedaços separados na tela
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_aberta, connectivity=8)
    
    if num_labels <= 1:
        return img_rgba
        
    # Identifica o ID da maior peça da tela (o nosso produto, ignorando o fundo)
    maior_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mascara_final = np.where(labels == maior_label, 255, 0).astype(np.uint8)
    
    # PASSO C: Suavização leve das bordas para o corte não ficar serrilhado
    mascara_suave = cv2.GaussianBlur(mascara_final, (3, 3), 0)
    
    # Aplica a máscara limpa na foto original
    img_array[:, :, 3] = cv2.bitwise_and(canal_alpha, canal_alpha, mask=mascara_final)
    
    return Image.fromarray(img_array)

# 4. Função de Recorte Combinada com Cache (IA + Filtro Dinâmico)
@st.cache_data(show_spinner=False)
def recortar_fundo(imagem_bytes, usar_alta_precisao, forca_desconexao):
    img = Image.open(io.BytesIO(imagem_bytes)).convert("RGBA")
    sessao_ia = carregar_modelo_ia()
    
    if usar_alta_precisao:
        recorte_bruto = remove(img, session=sessao_ia, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10)
    else:
        recorte_bruto = remove(img, session=sessao_ia)
        
    # Aplica a limpeza usando a força que você escolheu no Slider
    recorte_limpo = limpar_sujeiras_e_itens_encostados(recorte_bruto, forca_desconexao)
    return recorte_limpo

# --- BARRA LATERAL DE CONTROLES ---
st.sidebar.header("🛠️ Ajustes de Limpeza")
forca_sep = st.sidebar.slider("✂️ Força para Desgrudar Itens", min_value=0, max_value=60, value=15, step=5, help="Aumente esse valor se clipes, papéis ou sujeiras encostadas no produto não sumirem automaticamente.")
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
    
    with st.spinner("🤖 IA processando e separando itens encostados..."):
        # Executa o recorte passando o valor do slider
        img_sem_fundo = recortar_fundo(bytes_arquivo, modo_precisao, forca_sep)
        
        # Auto-crop ao redor do material
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

    st.success("✅ Imagem processada! Use o slider 'Força para Desgrudar Itens' à esquerda se algum item teimoso ainda aparecer.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(arquivo_enviado, caption="Foto Original", use_container_width=True)
    with col2:
        st.image(imagem_final, caption="Resultado Final Dm3 (Atualiza em tempo real!)", use_container_width=True)
        
    st.download_button(
        label="⬇️ Baixar Imagem Pronta",
        data=byte_im,
        file_name="produto_dm3_pronto.png",
        mime="image/png",
        use_container_width=True
    )