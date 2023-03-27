import os

import asrp
import nlp2

chunk_sec = 30


def hubert_layer9_code500():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin', './')
    hc = asrp.HubertCode("facebook/hubert-base-ls960", './hubert_base_ls960_L9_km500.bin', 9,
                         chunk_sec=chunk_sec,
                         worker=20)
    return hc

def hubert_layer6_code50():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin', './')
    os.rename('./km.bin', './hubert_base_ls960_L6_km50.bin')
    hc = asrp.HubertCode("facebook/hubert-base-ls960", './hubert_base_ls960_L6_km50.bin', 6,
                         chunk_sec=chunk_sec,
                         worker=20)
    return hc


def hubert_layer6_code100():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin', './')
    os.rename('./km.bin', './hubert_base_ls960_L6_km100.bin')
    hc = asrp.HubertCode("facebook/hubert-base-ls960", './hubert_base_ls960_L6_km100.bin', 6,
                         chunk_sec=chunk_sec,
                         worker=20)
    return hc


def hubert_layer6_code200():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km200/km.bin', './')
    os.rename('./km.bin', './hubert_base_ls960_L6_km200.bin')
    hc = asrp.HubertCode("facebook/hubert-base-ls960", './hubert_base_ls960_L6_km200.bin', 6,
                         chunk_sec=chunk_sec,
                         worker=20)
    return hc
