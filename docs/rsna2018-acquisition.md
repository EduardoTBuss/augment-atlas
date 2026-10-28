# Aquisição offline RSNA Pneumonia 2018

Implementa a definição aprovada por Eduardo em 28/10/2026 (C7). A ferramenta usa
somente arquivos locais, stdlib e Pillow; não usa pydicom e não faz downloads.
Execução real dos 30.000 DICOMs ainda depende do coordenador no servidor.

## Política fixada em config

- Grupo: os oito dígitos iniciais de `img_id` NIH; `subset_img_id` é UUID de imagem.
  Nunca é usado como paciente. Não há união entre pacientes.
- Somente o labelGroup `Calculated`: `Lung Opacity` = 1;
  `No Lung Opacity / Not Normal` e `Normal` = 0.
  Anotações repetidas da mesma classe (caixas) contam como um rótulo por imagem.
- Sem consenso: excluída com `no_consensus_label`, mesmo se outros grupos tiverem
  anotações. Sem SOPInstanceUID: resolve por StudyInstanceUID apenas se houver
  exatamente uma imagem candidata. Ambiguidade, rótulo desconhecido ou classes
  Calculated conflitantes bloqueiam a aquisição para revisão.
- Duplicatas: SHA256 dos 1.048.576 bytes **decodificados** em modo L, antes de
  reduzir a imagem. Todas as imagens de cada conjunto repetido são excluídas com
  `pixel_duplicate`, inclusive entre pacientes distintos. A busca abrange imagens
  sem consenso. Se houver dois motivos, `reasons` conserva ambos, com
  `no_consensus_label` como motivo principal.
- Derivado `rsna2018-derived-512-v1`: PNG L 8-bit 512×512, Pillow BILINEAR,
  sem janelamento, sem EXIF, ICC ou comentários. O leitor exige CR,
  MONOCHROME2, unsigned 8 bits, 1024×1024, um frame e transfer syntax
  JPEG baseline `1.2.840.10008.1.2.4.50`. ViewPosition PA/AP fica no manifesto privado.
- Licença/atribuição registrada:
  “RSNA Pneumonia Detection Challenge 2018 terms — attribution to NIH Clinical Center and RSNA”.

`configs/acquisition/rsna2018-v1.json` fixa os três SHA256 e todas essas escolhas.
`expected` exige 30.000 imagens, 29.684 rotuladas, 316 sem consenso, 3 conjuntos
de duplicatas com 6 imagens (todas rotuladas), 29.678 incluídas e 12.233 pacientes
incluídos. O número de pacientes fornecido era **aproximado**; continua sendo uma
barreira estrita, com essa proveniência explícita no config. A expectativa de
duplicatas veio do coordenador; o auditor de referência hasheava o payload
encapsulado, enquanto esta aquisição hasheia os pixels decodificados. Não há
alegação de validação desses totais reais nesta implementação.

## Comandos no servidor

No checkout com estes arquivos e com Python >= 3.11/Pillow já disponíveis:

```bash
cd ~/projetos/augment-atlas
PYTHONPATH=src python -m augment_atlas.acquisition.rsna2018 \
  --raw ~/projetos/augment-atlas/data/rsna2018/raw \
  --out ~/projetos/augment-atlas/data/rsna2018 \
  --config configs/acquisition/rsna2018-v1.json --workers 2

PYTHONPATH=src python -m augment_atlas.acquisition.rsna2018 \
  --raw ~/projetos/augment-atlas/data/rsna2018/raw \
  --out ~/projetos/augment-atlas/data/rsna2018 \
  --config configs/acquisition/rsna2018-v1.json verify
```

Sem argumento posicional, deriva o que falta e faz verify automaticamente.
`--workers 0` (padrão) roda inline; 1 ou 2 usa processos CPU com spawn. No máximo
`workers` jobs ficam em voo. Somente o pai escreve arquivos e ledger.
Não são criados pools/folds científicos neste passo.

Cada comando recalcula os SHA256 dos três arquivos de origem (inclusive do ZIP
inteiro); a retomada evita decodificar novamente membros DERIVED, mas a leitura
do ZIP para seu hash continua necessária. Preserve os arquivos raw para verify.

## Retomada, integridade e publicação

Reutiliza Ledger SQLite WAL/synchronous=FULL e ProcessLock do ISIC.
O ledger fixa identidade do config, fontes e versões Pillow/JPEG. A única chave
operacional excluída da identidade é `accepted_divergences`; workers é uma
opção CLI. Trocas de política, fontes, expected ou decoder exigem outra raiz de
aquisição; não altere os pins no SQLite.

Por imagem: PENDING → FETCHED (SHA256 do DICOM) → arquivo PNG temporário com fsync
→ rename + fsync do diretório → DERIVED com metadados/hashes. Um FETCHED é relido
do ZIP local após interrupção; um DERIVED é rehasheado no verify. Repetir o comando
retoma sem sobrescrever PNGs já DERIVED. SIGINT/SIGTERM retorna 75. O lock impede
dois escritores na mesma raiz. Em falta de espaço, erros de escrita encerram com
2; libere espaço e repita a aquisição. Não há download, spool de DICOM nem retry
automático para erros de integridade.

No verify: reconcilia novamente os JSONs com o ZIP, confere identidade e completude
do ledger, rehasheia todos os PNGs, calcula duplicatas pelos hashes de pixels no
ledger e recompõe os manifestos. Não decodifica novamente o JPEG. A proveniência
dos hashes decodificados vem da derivação e dos hashes de origem/decoder fixados.
Arquivos PNG extras e derivados ausentes/adulterados bloqueiam READY.

Artefatos são escritos por fsync + rename. READY vem por último, após o sumário
público passar o lint completo (inclusive metadados do próprio lint). Ao iniciar
uma execução/verificação sob lock, READY e sumário público anteriores são
invalidados. Uma falha não deixa um marcador válido antigo.

## Saídas e privacidade

Todos os caminhos abaixo são relativos a `--out`:

```text
acquisition-v1/{config.json,ledger.sqlite,lock}
derived-512-v1/images/<NIH img_id>.png
manifest/{manifest.csv,records.json,manifest-private.json,divergences.json,READY.json}
public/rsna2018-summary-public.json
```

**Somente o sumário público pode sair do servidor.** Raw, PNGs, ledger, logs e
todos os manifestos são privados, inclusive `manifest.csv` e `records.json`.
Todos os DICOMs são decodificados e seus PNGs mantidos para auditoria/reverificação;
as exclusões aplicam-se a `manifest.csv`/`records.json`, que contêm somente imagens
incluídas. Use sempre o manifesto para treino, nunca uma varredura do diretório.

`manifest.csv`: `path,group,label,sha256`, caminhos relativos ao root. Consumível
por `data.load_records` com `adapter=manifest_csv`, `require_image_hashes=true`
e o hash do manifesto em `READY.files["manifest.csv"]`.
O manifesto privado conserva os IDs de origem, rótulos, ViewPosition, status,
motivos, hashes de arquivo/pixels e grupos de duplicatas. O sumário contém
apenas agregados, atribuição, identidades e hashes; o lint verifica regexes
NIH/UUID/UID e todos os IDs conhecidos, inclusive o prefixo de paciente.

## Divergências e códigos de saída

- 0: READY publicado.
- 3: hash de origem/identidade divergente, ou contagens não aceitas.
- 2: falha de integridade, DICOM incompatível, erro de escrita ou privacy lint.
- 64: config inválido; opções CLI inválidas retornam 2 via argparse.
- 75: lock ocupado, fonte/ledger ausente, verify incompleto ou interrupção.

Contagens divergentes ficam em `manifest/divergences.json`. Uma decisão de Eduardo
pode ser registrada em `accepted_divergences`, com exatamente o campo e o valor
observado:

```json
{
  "field": "counts.patients_included",
  "observed": 123,
  "resolution": "EXEMPLO sintético: substituir pela decisão real e sua justificativa",
  "decided_by": "Eduardo"
}
```

O número 123 é apenas um exemplo, não uma aceitação de dados reais.
Rode verify novamente após registrar a decisão. Uma aceitação só corresponde ao
mesmo campo e observado; não aceita SHA256 divergente, conflitos de rótulo nem
DICOM incompatível. Não altere expected para contornar a identidade fixada.

## Testes

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_rsna2018_acquisition.py' -v
```

Fixtures inteiramente sintéticas: três DICOMs JPEG encapsulados com fragments e
sequência indefinida, mappings e export MD.ai. Cobrem grupos/labels/fallback,
exclusão sem consenso, todas as duplicatas (inclusive sem consenso), interpolação
com padrão espacial, saída sem metadados, adaptador de treino, hashes/contagens,
aceitação explícita, lint final, interrupção/retomada, idempotência, dois processos,
PNG adulterado e falhas de identidade/formato.
