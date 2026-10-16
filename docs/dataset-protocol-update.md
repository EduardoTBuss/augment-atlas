# Datasets e HPO econômico — atualização de16/10/2026

Direção aprovada: INbreast, retinopatia diabética, dermoscopia e raio-X do tórax;
dez braços, ResNet50, agrupamento por paciente e as métricas discutidas. Esta
atualização substitui a preferência anterior por HPO dentro de cada fold:
**uma busca sem augmentation por dataset, receita compartilhada pelos dez braços**.
Releases/tarefas abaixo são recomendações documentais do Codex para revisão.
Nenhuma nova base foi adquirida ou busca científica executada.

## Bases recomendadas e alternativas

| Modalidade | Recomendação do Codex | Alternativas e motivo da escolha |
|---|---|---|
| Retina | **mBRSET1.0**,5.164imagens/1.291pacientes, campo `patient`, DR ordinal0–4 | BRSET1.0.2:16.266/8.524, `patient_id`. EyePACS é referência maior com prefixo de sujeito documentado; APTOS/IDRiD mantêm uso recente, mas seu vínculo paciente não foi confirmado |
| Pele dermoscópica | **ISIC2020**,33.126imagens/2.056pacientes, `patient_id`, melanoma versus demais | É tarefa binária, diferente de HAM7classes/ISIC2019oito. Esses releases não têm vínculo paciente confirmado na documentação consultada. A escolha binária deve constar como emenda, sem herdar silenciosamente o endpoint multiclasse |
| Tórax | **CheXpert**,224.316radiografias/65.240pacientes, identidade pelo diretório de paciente | MIMIC-CXR-JPG tem `subject_id` e download parcial documentado. VinDr permanece referência contemporânea, mas remove patient ID e não sustenta CV por paciente a partir de image_id |

Fontes/schema: [mBRSET](https://physionet.org/content/mbrset/1.0/),
[BRSET1.0.2](https://physionet.org/content/brazilian-ophthalmological/1.0.2/),
[ISIC2020](https://doi.org/10.1038/s41597-021-00815-z),
[CheXpert](https://arxiv.org/abs/1901.07031),
[MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/),
[VinDr e de-identificação](https://physionet.org/content/vindr-cxr/1.0.0/).

Uso competitivo recente não é sinônimo de adequação do split: estudos de2026
usam [APTOS/IDRiD/Messidor2](https://doi.org/10.1038/s41467-026-70077-z) e
[CheXpert/MIMIC/VinDr](https://doi.org/10.1038/s41467-026-76076-4). O
[preprint de abril2026](https://doi.org/10.64898/2026.04.17.26351092) compara
foundation models em BRSET com avaliação externa mBRSET; essa evidência é de
preprint, e apenas seu resumo oficial foi acessível nesta consulta.
[PanDerm2025](https://doi.org/10.1038/s41591-025-03747-y) usa HAM10000.
Esses métodos/backbones não estabelecem desempenho esperado de uma comparação
controlada ResNet50. A pesquisa é limitada, sem ranking global de SOTA.

### Acesso e integridade

BRSET/mBRSET e MIMIC exigem credenciamento, treinamento e acordo PhysioNet.
CheXpert tem gate de acesso/termos. O tamanho de BRSET/mBRSET emGB não foi
verificado. Não presumir autorização de dados a partir da licença de código.

O [portal ISIC](https://challenge.isic-archive.com/data/) lista23GB para JPEG2020;
a [release StanfordAIMI512](https://huggingface.co/datasets/StanfordAIMI/CheXpert-v1.0-512)
lista38,3GB. Com aproximadamente41GiB livres e reserva operacional20GiB,
aquisição integral requer outro armazenamento ou estratégia parcial verificável.
A disponibilidade atual de um pacote oficial CheXpertSmall11GB não foi confirmada.
Não baixar tudo simultaneamente nem apagar resultados anteriores para caber.

ISIC2020 declara425duplicatas de pixels; reconciliar a lista oficial antes das
partições. mBRSET tem pequena divergência documental no número de rótulos DR
ausentes: resolver pelo release/hash adquirido e registrar exclusões. No BRSET,
grau4 inclui pós-laser. Patient IDs explícitos ainda exigem auditoria de nulos,
consistência, duplicatas e namespace; não preencher faltantes com IDs de imagem.

## Uma busca por base sem usar o teste na seleção

Escolha metodológica recomendada pelo Codex, apoiada pela revisão estatística:

1. Separar uma vez grupos de desenvolvimento `D`≈20% e avaliação `E`≈80%.
   Persistir listas e hashes, sem escolher seeds por score. Todas as visitas,
   olhos, lesões e estudos do mesmo paciente permanecem juntos.
2. Fazer **todo** HPO em `D`: candidatos treinam em `D_fit`, seleção/época/limiar
   usam `D_select`. Nenhum treino candidato nem escolha adaptativa consulta `E`.
3. Congelar receita, pré-processamento, número de épocas e limiares. A proposta
   de quatro receitas/uma seed/até20épocas por candidata ainda é orçamento de
   planejamento, sem execução científica autorizada por este documento.
4. Dividir `E` em cinco folds agrupados e fixos. Em cada fold, inicializar um
   modelo novo com ImageNet e treinar em `D ∪ (E menos o fold de teste)`, usando
   a receita e duração congeladas. Não escolher época novamente em `E`.
5. Todos os dez braços/três seeds usam os mesmos folds e receita. Somente `E`
   recebe predições finais fora da amostra; `D` nunca é teste. A reincorporação
   de `D` ao treino final é permitida após o congelamento, sem reutilizar o
   checkpoint candidato como inicialização final.

O [select-shuffle-test publicado](https://doi.org/10.1148/ryai.220232) existe e é
uma proposta distinta. Embaralhar a atribuição dos folds não desfaz a influência
dos rótulos já usados na seleção. Aqui adotamos a separação mais estrita acima;
não descrevemos shuffle como eliminação de viés. Ver também
[Cawley e Talbot](https://www.jmlr.org/papers/v11/cawley10a.html).

Com as proporções nominais, cada teste cobre≈16% da base e treino≈84%; não é
CV5 sobre100% da população. Transferir uma receita ajustada em `D` pequeno para
treinos maiores é uma limitação assumida, sem corrigi-la olhando os testes.

### Auditoria de viabilidade do INbreast existente, sem treino

Uma proposta privada nova preservou os folds anteriores e passou na auditoria
estrutural, usando os grupos atuais e seeds fixas17/29/31, sem retries de seeds:

| População | Imagens | Grupos | Negativas/positivas |
|---|---:|---:|---:|
| D |77|21|57/20|
| D_fit |59|17|43/16|
| D_select |18|4|14/4|
| E |310|82|230/80|

Testes emE:62/62/64/61/61imagens,16positivas cada; cobertura deE exatamente uma
vez e grupos separados. Apenas quatro grupos e quatro imagens positivas na
seleção tornam HPO instável; ambas as classes presentes não provam precisão.
Não chamar imagens positivas de pacientes positivos. Identidade grupo–paciente
do INbreast ainda não foi comprovada e o piloto já examinou a base.
Esta é viabilidade por grupos, sem HPO/scores ou certificação de pacientes.
Se D for inviável, a alternativa mais barata é receita externa fixa sem HPO,
ou aumentar D antes dos scores, aceitando menos avaliação.

## Métricas aceitas e fronteira da implementação

| Tarefa | Primária |
|---|---|
| INbreast binário | AUROC, com alvo/classe positiva explícitos |
| DR ordinal0–4 | QWK, pesos quadráticos e regra de classe congelada |
| Pele multiclasse | Macro-AUROC OvR |
| Pele binária, se a recomendaçãoISIC2020 for adotada | AUROC binária, emenda explícita de tarefa |
| Tórax multilabel | Macro-AUROC sobre rótulos predefinidos, com política de incerteza |

Secundárias:macro-F1,balanced accuracy eAUPRC quando pertinentes. Limiar fixo ou
escolhido só emD; classes ausentes geram métrica indefinida reportada, sem remoção
silenciosa. Agrupamento por paciente vale para split e incerteza; o alvo pode
continuar por imagem/lesão/estudo. Não usar fold×seed como pacientes independentes.

O trainer smoke atual calcula loss/accuracy de classes exclusivas. O serviço
HPO global, exportação de predições/novas métricas e treino multilabel ainda
precisam ser implementados e revisados; não converter doenças em classe única.
`selection-only` atualmente exige `within-outer-fold`: não afrouxar essa guarda
sem contratos que comprovem os pools globais separados. As fases científicas
continuam bloqueadas. Testes pequenos na GPU validam infraestrutura.
