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

# =========================================================
# CONSTANTES DE CONFIGURAÇÃO
# (antes espalhadas pelo código como "números mágicos")
# =========================================================
LIMIAR_ALPHA_MASCARA = 20          # binarização da máscara de recorte
LIMIAR_ALPHA_CORTE_OTICO = 35      # ignora sombras/ruído quase invisível
ALPHA_MATTING_FG = 240             # alta precisão: primeiro plano
ALPHA_MATTING_BG = 10              # alta precisão: fundo
CACHE_TTL_SEGUNDOS = 600           # resultados processados expiram em 10 min
MARGEM_SEGURANCA_PROCESSAMENTO = 1.3  # margem sobre o tamanho do fundo

# Config da página
st.set_page_config(page_title="Gerador Dm3 - Pro", page_icon="📦", layout="wide")
st.title("📦 Dm3 - Estúdio de Fotos Automático (Foco em Produtos)")
st.write("Suba a foto do material. O sistema isola, limpa itens encostados e centraliza automaticamente a mercadoria.")

# 1. Carregar o fundo padrão (Gabarito)
try:
    fundo_padrao = Image.open("fundo_dm3.png").convert("RGBA")
except FileNotFoundError:
    st.error("⚠️ Erro: O arquivo 'fundo_dm3.png' não foi encontrado na pasta do sistema!")
    st.stop()

largura_fundo, altura_fundo = fundo_padrao.size

# Como a imagem final SEMPRE é redimensionada para caber no fundo, não faz
# sentido rodar a IA numa foto de 4000x3000px vindas de celular: isso é o
# maior consumidor de RAM/tempo do app. Calculamos aqui o maior lado que
# realmente precisamos processar, com uma margem de segurança.
LIMITE_LADO_PROCESSAMENTO = int(max(largura_fundo, altura_fundo) * MARGEM_SEGURANCA_PROCESSAMENTO)


# 2. Carregar Inteligência Artificial COM TRAVA DE MEMÓRIA
# (max_entries=1 impede ter 2 IAs na RAM ao mesmo tempo!)
@st.cache_resource(max_entries=1, show_spinner="Carregando modelo de IA...")
def carregar_modelo_ia(modo_roupa=False):
    try:
        if modo_roupa:
            # IA especialista em moda: isola tecidos e ignora pescoço/corpo de manequins
            return new_session("u2net_cloth_seg")
        else:
            # IA geral: excelente para caixas, totens, ferramentas e produtos em geral
            return new_session("isnet-general-use")
    except Exception as erro:
        st.warning(
            f"⚠️ Não consegui carregar o modelo principal ({erro}). "
            "Usando o modelo leve 'u2netp' como alternativa — a qualidade do "
            "recorte pode ficar um pouco inferior."
        )
        return new_session("u2netp")


def redimensionar_para_processamento(imagem_bytes, limite_lado):
    """
    Reduz a imagem enviada para o maior tamanho realmente necessário antes
    de passar pela IA. Isso é o principal ganho de RAM: uma foto de celular
    de 12MP processada a 4000px consome muito mais memória (e tempo) do que
    processá-la já no tamanho final que será usado no fundo.
    """
    img = Image.open(io.BytesIO(imagem_bytes)).convert("RGBA")

    if max(img.size) > limite_lado:
        img.thumbnail((limite_lado, limite_lado), Image.Resampling.LANCZOS)

    return img


# 3. FUNÇÃO INTELIGENTE: Remove itens encostados e "fantasmas" de fundo
def limpar_itens_encostados(img_rgba, forca_corte):
    img_array = np.array(img_rgba)
    canal_alpha = img_array[:, :, 3]

    # Cria uma máscara binária pura (também elimina ruído de fundo com
    # transparência residual, tipo sombras/reflexos que a IA não zerou)
    _, binaria = cv2.threshold(canal_alpha, LIMIAR_ALPHA_MASCARA, 255, cv2.THRESH_BINARY)

    if forca_corte > 0:
        k_size = int(forca_corte)
        if k_size % 2 == 0:
            k_size += 1
        elemento_estrutura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        # Erosão temporária para quebrar pontes finas (arames de clipes, post-its)
        mascara_trabalho = cv2.erode(binaria, elemento_estrutura, iterations=1)
    else:
        elemento_estrutura = None
        mascara_trabalho = binaria

    # SEMPRE mantém apenas o maior bloco conectado da imagem — isto é o que
    # elimina "fantasmas" de fundo (móveis, sombras, objetos ao fundo que a
    # IA isolou como blocos separados do produto). Antes isso só rodava se
    # o usuário mexesse no slider; agora roda por padrão em toda imagem.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara_trabalho, connectivity=8)

    if num_labels > 1:
        maior_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        nucleo_produto = np.where(labels == maior_label, 255, 0).astype(np.uint8)
    else:
        nucleo_produto = mascara_trabalho

    if elemento_estrutura is not None:
        # Dilata o núcleo de volta para devolver a borda original e lisa do produto
        mascara_restaurada = cv2.dilate(nucleo_produto, elemento_estrutura, iterations=1)
    else:
        mascara_restaurada = nucleo_produto

    alpha_final = cv2.bitwise_and(canal_alpha, mascara_restaurada)

    # Libera arrays intermediários explicitamente (ajuda em imagens grandes)
    del mascara_trabalho, nucleo_produto, mascara_restaurada, labels, stats, binaria

    alpha_suave = cv2.GaussianBlur(alpha_final, (3, 3), 0)
    img_array[:, :, 3] = alpha_suave

    resultado = Image.fromarray(img_array)
    del img_array, canal_alpha, alpha_final, alpha_suave
    return resultado


# 4. Função de Recorte com TRAVA DE MEMÓRIA
# (max_entries=2 impede o servidor de cair; ttl expira resultados antigos)
@st.cache_data(max_entries=2, ttl=CACHE_TTL_SEGUNDOS, show_spinner=False)
def recortar_fundo(imagem_bytes, usar_alta_precisao, forca_desconexao, modo_roupa, limite_lado):
    img = redimensionar_para_processamento(imagem_bytes, limite_lado)
    sessao_ia = carregar_modelo_ia(modo_roupa)

    if usar_alta_precisao:
        recorte_bruto = remove(
            img,
            session=sessao_ia,
            alpha_matting=True,
            alpha_matting_foreground_threshold=ALPHA_MATTING_FG,
            alpha_matting_background_threshold=ALPHA_MATTING_BG,
        )
    else:
        recorte_bruto = remove(img, session=sessao_ia)

    recorte_limpo = limpar_itens_encostados(recorte_bruto, forca_desconexao)

    del img, recorte_bruto
    return recorte_limpo


# --- BARRA LATERAL DE CONTROLES ---
st.sidebar.header("🚀 Desempenho e RAM")

if st.sidebar.button("🧹 Limpar Memória RAM", use_container_width=True,
                      help="Clique aqui se o sistema estiver rodando há muito tempo e parecer um pouco lento."):
    st.cache_data.clear()
    st.cache_resource.clear()
    gc.collect()
    st.sidebar.success("Memória do servidor limpa!")

st.sidebar.caption(
    f"📏 Fotos maiores que {LIMITE_LADO_PROCESSAMENTO}px de lado são "
    "reduzidas automaticamente antes do processamento para economizar RAM, "
    "sem perda perceptível de qualidade no resultado final."
)

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
    min_value=0, max_value=40, value=0, step=2,
    help="O sistema já remove fundos 'fantasmas' automaticamente. Use este slider só quando algo ainda estiver ENCOSTADO no produto (clipe, papel) e for confundido como parte dele — aumente gradualmente até eles sumirem."
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

    try:
        with st.spinner("🤖 IA processando, isolando e centralizando o produto..."):
            img_sem_fundo = recortar_fundo(
                bytes_arquivo, modo_precisao, forca_sep, modo_roupa, LIMITE_LADO_PROCESSAMENTO
            )

            # --- CORTE ÓPTICO ANTI-FANTASMA ---
            img_arr = np.array(img_sem_fundo)
            alpha_canal = img_arr[:, :, 3]
            y_indices, x_indices = np.where(alpha_canal > LIMIAR_ALPHA_CORTE_OTICO)

            if len(x_indices) > 0 and len(y_indices) > 0:
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                img_cortada = img_sem_fundo.crop((x_min, y_min, x_max + 1, y_max + 1))
            else:
                img_cortada = img_sem_fundo

            fator_escala = escala_manual / 100.0
            proporcao = min(
                (largura_fundo * fator_escala) / img_cortada.width,
                (altura_fundo * fator_escala) / img_cortada.height,
            )
            nova_largura = int(img_cortada.width * proporcao)
            nova_altura = int(img_cortada.height * proporcao)
            img_redimensionada = img_cortada.resize((nova_largura, nova_altura), Image.Resampling.LANCZOS)

            # --- CÁLCULO DE CENTRALIZAÇÃO AUTOMÁTICA ---
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
                del arr_redim
            else:
                posicao_x = ((largura_fundo - nova_largura) // 2) + ajuste_x
                posicao_y = ((altura_fundo - nova_altura) // 2) + ajuste_y

            imagem_final = fundo_padrao.copy()
            imagem_final.paste(img_redimensionada, (posicao_x, posicao_y), img_redimensionada)

            buf = io.BytesIO()
            imagem_final.save(buf, format="PNG")
            byte_im = buf.getvalue()

            # Libera tudo que não precisamos mais antes de desenhar na tela
            del img_arr, alpha_canal, y_indices, x_indices, img_cortada, img_redimensionada, buf
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
            use_container_width=True,
        )

    except Exception as erro:
        st.error(
            "❌ Não consegui processar essa imagem. Tente outra foto ou clique em "
            "'Limpar Memória RAM' na barra lateral e tente novamente."
        )
        st.caption(f"Detalhe técnico: {erro}")
        gc.collect()

    finally:
        # Independentemente de sucesso ou erro, garante que a RAM
        # não fique presa com referências do processamento desta imagem
        del bytes_arquivo
        gc.collect()
