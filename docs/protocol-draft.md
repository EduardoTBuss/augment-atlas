# Proposta de protocolo — 16/10/2026

Status: **rascunho para decisão conjunta**. Os quatro braços isolados foram
solicitados por Eduardo; esta proposta não aprova automaticamente datasets,
faixas, HPO, orçamento ou a campanha científica. Os exemplos executáveis seguem
`smoke/provisional` e preservam o algoritmo anterior.

Decisões posteriores de Eduardo em16/10: HPO interno por fold sem augmentation;
parâmetros candidatos abaixo mantidos para os testes de hoje; até8h de engenharia
e até10jobs de2épocas, umaGPU, teto30min de treinamento. Escolhas técnicas
provisórias feitas pelo Codex estão identificadas para revisão do projeto.
A campanha600 e os releases/tarefas finais não foram iniciados.

## Pergunta e dez braços

Medir como gamma e CLAHE alteram uma receita ResNet-50 ImageNet FP32 congelada,
com e sem augmentation básico. Não partir da premissa de que o básico ajuda ou
prejudica todas as tarefas. O desenho é um fatorial 2 × 5:

| Transformação adicional | Sem básico | Com básico |
|---|---|---|
| Nenhuma | `none` | `basic` |
| Gamma leve | `gamma_light` | `basic_gamma_light` |
| Gamma forte | `gamma_heavy` | `basic_gamma_heavy` |
| CLAHE leve | `clahe_light` | `basic_clahe_light` |
| CLAHE forte | `clahe_heavy` | `basic_clahe_heavy` |

Não há combinação gamma + CLAHE. Para uma transformação T e métrica M, estimar:
efeito isolado `M(T) − M(none)`; incremento sobre básico
`M(basic+T) − M(basic)`; interação entre básico e T, diferença entre esses dois
incrementos. Manter mesma inicialização, partições, ordem de amostras, receita
e orçamento de atualizações entre braços. A nova candidata usa sorteios com
identidade por operação para emparelhar doses de T nos braços isolados e
combinados; o backend sequencial legado é preservado, com sua limitação.

Com quatro bases, cinco folds e três seeds: **600 treinos finais**, antes de
HPO/falhas, contra os 360 do desenho anterior. Uma primeira seed cobriria 200,
não 120. Esses totais não são uma fila autorizada a executar.

## Interpretação da literatura

A leitura do [postprint de Dicu e colaboradores](https://hdl.handle.net/10651/83440)
(HAIS2025; online2025, citação editorial2026) mostrou que CLAHE elevou AP50 do
Faster R-CNN de77.90 a91.35, mas reduziu AP75 de33.00 a8.50. Flip/rotação tiveram
quedas nesse protocolo de detecção DDSM. Isso não demonstra uma regra sobre
classificação INbreast nem ganho universal de CLAHE. Em contraponto,
[MAMMO](https://arxiv.org/abs/1811.02661) relata benefício de flips na classificação
multivista em outra coorte. Tarefa, aquisição, intensidade, métrica e partição
limitam a transferência dessas conclusões.
Os braços isolados são úteis justamente porque permitem testar a interação.

A pesquisa registrada contém ranges usados em tarefas diferentes, divergências
entre tabela/código e valores sem ablação. Ausência de um ótimo publicado não
impede uma escolha exploratória explícita; não a transforma em ótimo validado.
Ver [pesquisa primária](dataset-research.md).

## Bases, tarefas e partições

Direção: INbreast, retinopatia diabética, dermoscopia ISIC e raio-X do tórax.
O interesse é uso recente e viabilidade, sem confundir tamanho com qualidade.

| Modalidade | Candidatos | Condição antes de congelar |
|---|---|---|
| Mamografia | INbreast já existente | Auditar elegibilidade e significado de `group_key`; alvo BI-RADS definido, sem chamar malignidade comprovada. Já explorado no piloto |
| Retina | APTOS/IDRiD versus EyePACS | Release/rótulos, olhos do mesmo paciente, duplicatas e identificador de agrupamento. DR ordinal em cinco graus é diferente de binarização |
| Pele | ISIC dermoscopia prioritário, HAM menor | Fixar release, classes e agrupamento disponível. HAM integra ISIC2019; não é seu teste externo independente. ISIC2024 total-body crops têm outra aquisição |
| Tórax | VinDr versus CheXpert/MIMIC | Acesso, paciente/estudo/vista, rótulos incertos e tarefa multilabel. Não forçar exclusividade de patologias para caber no trainer atual |

Recomendação para INbreast pequeno: cinco folds externos agrupados, com
distribuição de classes auditada; seleção interna separada dentro do treino de
cada fold. **HPO interno limitado**, sem augmentation, escolhido por Eduardo.
Uma busca global exigiria reservar
grupos de desenvolvimento fora de TODOS os testes e reduz a avaliação disponível.
Uma única seleção interna não é CV interna completa. Ver
[CV aninhada](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
e [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

Salvar listas efetivas, versões e hashes antes do ajuste; mesmas partições nas
três seeds. Não usar cinco hashes/holdouts independentes como cinco folds. O
`group-hash-v1` atual continua apenas técnico. A análise principal usa o alvo
clínico por imagem/lesão/estudo e estima incerteza com agrupamento apropriado;
split por paciente não exige um único rótulo por paciente.

## Receita, HPO e métricas propostas

ResNet-50 com pesos ImageNet verificados por hash; FP32, autocast/TF32 desligados,
resize e normalização congelados. Propor 224px como ponto inicial de viabilidade,
sem assumir que essa resolução preserva todas as lesões de mamografia. A escolha
da resolução exige inspeção em desenvolvimento e orçamento próprio. Random-init
smoke não substitui a validação do caminho pré-treinado.

HPO escolhido: pesquisar apenas treino sem augmentation, com mesma grade e
limite por fold; congelar receita antes de executar os dez braços. Proposta
inicial para discussão: quatro receitas candidatas, uma seed de busca e teto de
20 épocas por candidata, dentro da seleção interna. Seleção por loss; métricas
externas nunca escolhem receita, intensidade ou stopping. Batch efetivo,
tratamento de BatchNorm e scheduler precisam ser registrados por receita.
Essa busca limitada estima uma receita sob esse orçamento, não um ótimo geral.

Primárias sugeridas por tarefa: AUROC para INbreast binário; QWK para DR ordinal;
macro-AUROC para pele multiclasse e tórax multilabel, com macro-F1/balanced
accuracy/AUPRC como secundárias pertinentes. Fixar classe positiva, tratamento
de classe ausente, rótulos incertos, ponderação e agregação antes de executar.
Salvar predições depois da seleção; não usar teste por época.

Proposta estatística mínima da revisão: efeitos médios de família gamma/CLAHE
e duas interações família × básico como quatro contrastes primários por base,
com pesos iguais predefinidos sobre intensidades. Reportar também efeitos por
intensidade, pois médias podem cancelar sinais opostos. Se o objetivo principal
for afirmar benefício/prejuízo de cada dose, promover esses contrastes antes dos
scores e ampliar a correção de multiplicidade. Não tratar folds/seeds como
pacientes independentes nem comparar AUCs brutas de bases distintas em um ranking.
INbreast continua exploratório pelo histórico do piloto; novas seeds não apagam
essa exposição. Garantir ausência de vazamento não desfaz exploração anterior.

## Parâmetros candidatos: proposta, não configuração científica aprovada

A proposta restritiva discutida com Claude pode servir de ponto de partida:

- Básico: flip H `p=0.5`, sem flip V; rotação ±15°, escala 0.8–1.2,
  translação ±10%, brilho/contraste ±0.2, sem hue/saturação. Esses valores não
  foram validados conjuntamente nas quatro modalidades. As probabilidades e
  ordem de geometria/fotometria ainda precisam ser fechadas. O básico atual do
  smoke só tem flip, brilho e contraste; não contém essas operações geométricas.
- Gamma: expoente direto em `[0,1]`, faixa leve 0.8–1.2 e forte 0.67–1.5,
  `p=0.3`, sem normalização por extremos por imagem. Valores propostos; não
  equivalem à fórmula extrema-preservada nem às probabilidades do smoke atual.
- CLAHE: candidato OpenCV4.12.0.88, clip fixo 2/4, grid 8×8, após resize, cinza ou
  luminância L de Lab em RGB, `p=0.3`. Fixar versão/conversão/dtype e ordem com
  o básico. Clip2 e4 não provam níveis universais leve/forte. O ambiente atual
  usa scikit-image e outra escala de clip. A imagem candidata acrescenta somente
  OpenCV ao ambiente existente, sem atualizar a pilha de treinamento.

Antes de usar essas propostas em treino científico: resolver probabilidades,
ordem, bordas/padding, preservação de lesões/orientação e distribuição de gamma;
gerar galeria SOMENTE em desenvolvimento, verificar numericamente operações e
congelar a versão. Uma imagem visualmente plausível não valida utilidade clínica.
Não alterar silenciosamente os seis braços legados ou reinterpretar seus scores.

## Fase noturna proposta e critérios de parada

Primeiro, validar os dez braços na candidata separada: uma GPU, uma seed, um
split técnico, até dez jobs, duas épocas por job, batch2. A resolução recomendada
para este ensaio é224px: em64px/grid8×8 os clips2/4 podem colapsar ao mesmo limite
inteiro de histograma. Essa escolha é do Codex para revisão posterior; não fixa
a resolução científica final. Teto aprovado de30minutos de
containers de treinamento, zero retry automático, sem testes científicos
protegidos/HPO. Dados sintéticos ou subset privado já usado como smoke; nunca
reutilizar resultados como confirmação científica. Não é necessário ocupar
a noite inteira se os checks terminarem antes.

Separadamente, o trabalho de engenharia pode continuar por até oito horas:
cache ImageNet com hash e montagem read-only, contrato RGB/CLAHE, splits
agrupados persistidos, métricas/predições e configuração da busca. Treino
adicional depende de manifesto com limite aceito, preflight e revisão. Nenhum
download de dataset com termos/credenciamento pendentes. Nenhuma campanha600.

Antes de carga sem acompanhamento humano, monitor independente do Git, teto
de temperatura, orçamento de tempo e disco e parada somente dos containers
deste ensaio. A checagem atual observou cerca de46GB livres no filesystem do
projeto; checkpoints e bases grandes exigem plano de armazenamento. Não apagar
resultados antigos para abrir espaço. A coleta síncrona atual, documentada em
[telemetria](telemetry.md), não é proteção térmica contínua.

Pendências para revisão: orçamento/grade exatos de HPO, básico geométrico,
endpoints/contrastes e protocolo estatístico. Datasets finais continuam
condicionados à auditoria de grupos, acesso e tarefa. Os testes não alteram
automaticamente essas decisões com base no score observado.
