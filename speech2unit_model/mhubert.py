import asrp
import nlp2

chunk_sec = 30


def mhubert_layer11_code1000():
    nlp2.download_file(
        'https://huggingface.co/voidful/mhubert-base/resolve/main/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin', './')
    hc = asrp.HubertCode("voidful/mhubert-base", './mhubert_base_vp_en_es_fr_it3_L11_km1000.bin', 11,
                         chunk_sec=chunk_sec,
                         worker=20)
    return hc
