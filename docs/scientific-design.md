# AugmentAtlas — desenho científico proposto

Data: 2026-10-15. Estado: **proposta metodológica; decisões científicas pendentes**.
O smoke atual valida infraestrutura. Nenhum treino, HPO ou análise de resultados foi
executado nesta revisão; nenhum parâmetro científico recebeu aprovação aqui.

Contexto: requisitos do projeto de 15/10/2026 e revisão estática dos contratos.
Este documento registra recomendações; decisões científicas continuam pendentes.

Atualização16/10: Eduardo escolheu HPO interno por fold, sem augmentation, e
solicitou quatro braços isolados adicionais. O desenho agora é2×5, dez braços;
detalhes, limites aceitos e escolhas provisórias do Codex em
[protocol-draft.md](protocol-draft.md). Faixas para os testes de hoje não são
ótimos nem aprovação da campanha. As alternativas históricas abaixo continuam
documentadas para explicar a escolha.

## Pergunta, unidades e limites da conclusão

A pergunta proposta é quanto cada política de augmentation altera o desempenho de
uma receita ResNet50 ImageNet FP32 congelada, em uma tarefa de classificação definida,
comparada aos controles. A direção atual contém INbreast, retina, dermoscopia e
raio-X do tórax, com releases finais a escolher. Os dez braços são `none`, `basic`,
`basic_gamma_light`, `basic_gamma_heavy`, `basic_clahe_light`, `basic_clahe_heavy`,
`gamma_light`, `gamma_heavy`, `clahe_light` e `clahe_heavy`.

O paciente é a unidade de agrupamento das partições quando identificável. Imagens, exames,
lesões e visitas do mesmo paciente permanecem juntos; se houver dependência maior
(por exemplo, duplicação entre fontes), usar a unidade superior. A origem e a
semântica de `group_key` precisam ser verificadas. Agrupar a divisão por paciente não
transforma automaticamente o alvo em um rótulo por paciente: lesões ou estudos do
mesmo paciente podem ter classes distintas. Definir alvo, agregação e ponderação
antes da avaliação; preservar agrupamento também na estimação de incerteza.
Sem identificador adequado, não
afirmar avaliação independente por paciente.

O estimando proposto por dataset e contraste é a diferença absoluta da métrica
primária entre políticas, mantendo fixos dados elegíveis, receita, divisões e
orçamento de treino. Rótulos, tarefa binária/multiclasse, métrica primária,
direção de benefício e diferença mínima relevante ainda serão escolhidos.
Não escolher a métrica que mais favorece o resultado observado. Métricas como
AUROC e sensibilidade em ponto operacional são exemplos, não endpoints aprovados.
Uma média de métricas entre seeds e a métrica de um ensemble de seeds são estimandos
diferentes; propor a primeira para comparar políticas de treino individuais.

As quatro bases são estudos de caso, não quatro réplicas aleatórias de todas as
modalidades médicas. Dataset, modalidade, população, tamanho e aquisição podem
estar confundidos; uma interação entre datasets não identifica efeito causal da
modalidade. A conclusão sobre augmentation é condicionada à arquitetura e à
receita escolhidas. HPO apenas sem augmentation estima o benefício ao adicionar
augmentation a essa receita, não o melhor desempenho possível de cada braço.

## Separar exploração e confirmação

O INbreast já examinado permanece piloto/adaptativo. Trocar folds ou seeds não
desfaz essa exposição. Registrar quais grupos, rótulos, métricas e imagens foram
vistos e quais escolhas isso influenciou. Novos testes confirmatórios exigem dados
realmente reservados ou independentes; sem eles, declarar avaliação interna
exploratória e suas limitações, mesmo que se use validação aninhada.

Antes de observar avaliações finais, versionar: elegibilidade, deduplicação,
rótulos, endpoints, agregação por paciente/fold/seed, exclusões, tratamento de
falhas, contrastes, multiplicidade e critérios de parada. Qualquer mudança depois
disso deve aparecer como emenda datada ou análise pós-hoc, preservando a análise
original. Sensibilidades não substituem silenciosamente o endpoint primário.

## HPO: escolha de16/10 e alternativa histórica

| Opção | Separação necessária | Condição de viabilidade / limite |
| --- | --- | --- |
| Um estudo por dataset em desenvolvimento separado | Todo HPO e suas validações usam população de desenvolvimento excluída de todos os testes finais; congelar receita antes deles | Alternativa histórica, não escolhida em16/10. Exige pacientes/eventos suficientes após reservar desenvolvimento. |
| HPO dentro de cada treino externo | Cada fold externo executa HPO apenas em suas partições internas agrupadas; seleção de época/limiar fica interna; teste externo nunca informa escolhas | Escolhido por Eduardo em16/10. A receita pode diferir entre folds, mas é compartilhada pelos dez braços dentro do fold; avalia um procedimento de seleção, não uma única receita global. |

Não escolher o vencedor por comparar resultados externos dessas duas opções.
Em ambas, HPO é sem augmentation; algoritmo, espaço de busca, número de tentativas,
seeds, limite de tempo, métrica e política de retomada precisam ser aprovados e
congelados. A seleção de época e limiar não pode consultar o teste. Estatísticas
de normalização, balanceamento e qualquer transformação aprendida usam somente
treino. Definir se o modelo final mantém um subconjunto de seleção ou é retreinado
por um número de épocas decidido internamente; ambas são escolhas prévias.

HPO aninhado evita acesso direto ao teste externo durante aquela execução, mas
não apaga escolhas humanas anteriores feitas com o mesmo dataset. Se não houver
população não exposta, essa limitação acompanha qualquer das opções.

## Menor desenho discriminante e orçamento limitado

1. Primeiro concluir um smoke sintético/manifesto privado autorizado: leitura,
   canais, rótulos, isolamento dos grupos, FP32, seleção interna, checkpoints,
   telemetria, persistência e retomada. Sucesso aqui é técnico; acurácia não é
   critério de sucesso científico.
2. Antes da matriz completa, fazer auditoria de viabilidade sem treinar: grupos e
   eventos por classe em cada partição, identidade dos pacientes, exposição
   anterior, duplicatas e precisão esperada para uma diferença mínima relevante
   escolhida cientificamente. Se eventos forem insuficientes, decidir reduzir
   folds ou rever a pergunta antes de gerar resultados. Não procurar split favorável.
3. No desenho dez braços, separar efeito isolado, incremento sobre básico e
   interação. Família de contrastes e multiplicidade ainda precisa ser congelada;
   a proposta mínima revisada está em protocol-draft.md. Não herdar automaticamente
   os cinco contrastes do desenho anterior.
4. A direção futura é4datasets ×10braços ×5folds externos ×3seeds = **600
   treinos de avaliação**, com os mesmos folds e seeds correspondentes por braço.
   Isso exclui HPO, piloto e retreinos: seus custos devem ser orçados separadamente.
   Não autoriza executar essa matriz agora. Se a pergunta científica for reduzida,
   reduzir a matriz e registrar a nova pergunta antes da execução.

Usar a mesma inicialização correspondente e listas de pacientes por contraste;
separar RNG de split, inicialização, amostragem e augmentation para evitar que uma
operação extra altere indevidamente as demais fontes de aleatoriedade. Comparar
com o mesmo limite de épocas e regras de seleção/parada. Fixar os ranges, ordem,
probabilidade e semântica dos canais de cada intervenção antes dos resultados;
os valores do smoke são provisórios. Transformações de avaliação devem ser
determinísticas e iguais entre braços. Não ajustar ranges no teste.

Critérios prévios de aceitação: nenhum paciente cruza partições; todos os braços
usam os mesmos testes; seleção/limiar/HPO sem teste; nenhuma combinação desaparece;
artefatos têm hashes; estimativa pareada e incerteza são reportadas. Uma alegação
de benefício exige a direção e magnitude mínima pré-especificadas e incerteza
compatível com a alegação. Efeito impreciso significa inconclusivo. Equivalência
exige margem própria e intervalo suficientemente estreito; ausência de
significância não demonstra ausência de efeito. Violação de isolamento invalida
a conclusão confirmatória correspondente. Falha técnica leva a diagnóstico e
emenda, não troca oportunista do split ou encerramento por resultado favorável.

## Inferência pareada, dependência e multiplicidade

Guardar predições individuais fora da amostra, ID pseudônimo de paciente, rótulo,
braço, fold, seed, checkpoint e motivo de qualquer ausência. Reconstruir métricas
dessas linhas e preservar também todas as execuções com falha. Nunca tratar
imagens do mesmo paciente ou os 15 valores fold × seed como 15 observações
independentes: há medidas repetidas e treinos externos sobrepostos.

Para um teste independente reservado, uma opção é bootstrap pareado dos pacientes
(ou da unidade superior), reutilizando exatamente os índices entre braços e
recomputando a diferença a cada amostra. Definir antes como lidar com classe
ausente nas reamostragens e métricas indefinidas. Reportar diferença absoluta,
intervalo de confiança e número de pacientes/eventos; isso estima incerteza
amostral condicional aos modelos treinados. Com seeds, calcular primeiro o
contraste correspondente por seed e usar o agregador registrado; mostrar também
a dispersão entre seeds. Três seeds não sustentam afirmações precisas sobre toda
a distribuição de treinamentos futuros.

Em validação cruzada, registrar previamente a agregação. Para métricas não lineares,
uma média de métricas por fold e uma métrica sobre predições OOF concatenadas podem
diferir; a última pode comparar escores de modelos com calibrações distintas.
Se usar bootstrap de pacientes sobre predições OOF fixas, preservar o pareamento
e os folds, mas identificar o intervalo como aproximação condicional à partição
e aos modelos. Ele não reproduz a incerteza completa do procedimento de treino;
o compartilhamento de treino entre folds impede interpretá-lo automaticamente
como teste confirmatório da diferença entre algoritmos. Não usar teste t comum
dos 15 escores. Inferência sobre o procedimento completo requer desenho e custo
adicionais ou teste independente; não lançar retreinos/sweeps para isso nesta revisão.

Se os cinco contrastes em quatro datasets forem todos confirmatórios, existe uma
família de **20 hipóteses**. Propor controle familiar pré-especificado (por exemplo,
Holm sobre p-valores marginais válidos); correção de multiplicidade não conserta
p-valores produzidos ignorando dependência. Intervalos pontuais de 95% não são
simultâneos: rotulá-los e definir intervalos simultâneos quando necessários à
alegação. Outra escolha é registrar um único contraste primário e tratar os demais
como exploratórios. A escolha é pendente. Não promover o braço vencedor de uma
exploração para hipótese confirmada nos mesmos dados.

## Manifesto da execução futura e falhas

Seguir `$scientific-run-manifest`: ID do estudo/protocolo; commit/branch e estado
sujo (ou snapshot com hash se não houver Git); comando, diretório, início/fim e
exit code; SO/runtime/dependências; fonte/versão/hash dos dados; elegibilidade,
pré-processamento e mapeamento de rótulos; listas efetivas agrupadas e hashes;
matriz completa, seeds distintos, hardware, FP32 e limites; origem/hash do HPO e
receita; definições/agregações das métricas; artefatos e hashes; limitações,
exclusões e status de cada combinação planejada. Nunca incluir identificadores
pessoais ou conteúdo de imagens nos artefatos públicos.

Antes da execução, fixar máximo de épocas, tentativas HPO, duração por tarefa,
limite total de GPU e número de retries. O runner recebe apenas esse manifesto
limitado via maestro. Distinguir falha de infraestrutura de divergência do método;
preservar ambas e repetir somente pela regra registrada, sem novas seeds escolhidas
pelo desempenho. Não substituir NaNs, ausências ou falhas silenciosamente, nem
comparar só o subconjunto de seeds que favorece um braço. Reportar completude e
sensibilidade a resultados ausentes; não há ainda ensaios brutos para auditar.

## Evidência do código atual e próxima decisão

Leitura estática em 15/10/2026: `contracts.py` exige `phase=smoke` e
`scientific_status=provisional`, receita técnica congelada e proveniência de HPO
pendente/externo. `data.py` persiste listas e hashes, verifica cobertura dos
registros e separação de grupos. Seu `group-hash-v1` produz uma partição técnica
aproximada 60/20/20, sem estratificação; `fold` altera o hash, não constrói uma
partição externa de cinco folds. Não usar esses cinco valores como CV exaustiva.
O código não atesta que `group_key` identifica paciente nem garante presença de
todas as classes em cada subconjunto. São limites esperados do smoke, não evidência
de validação científica concluída. As configurações em `configs/smoke/` são
exemplos técnicos provisórios.

Atualização16/10: cinco folds externos e um holdout interno por fold foram
preparados e auditados sobre o manifesto privado existente:387imagens,
103grupos, sem interseção de grupos entre partições, ambas as classes em cada
partição e cobertura externa de cada imagem exatamente uma vez. A origem atual
de `group_key` é um token do nome de aquisição; sua correspondência com paciente
ainda não foi comprovada. Portanto esses folds não são certificados como
independentes por paciente. Nenhum modelo/HPO foi ajustado nessa auditoria.

O contrato `selection-only` foi implementado para preparar um objetivo interno
sem inferência/métricas no teste externo. Checagens de integridade dos arquivos
continuam incluindo o manifesto completo. O serviço de busca não está implementado.

Próximo passo do protocolo: resolver identidade dos grupos, exposição anterior,
tarefas e endpoints; congelar espaço/orçamento da busca interna escolhida e as
intervenções antes de qualquer avaliação final. A discussão prioriza retina, dermoscopia (ISIC com HAM
como alternativa menor) e raio-X do tórax, além do INbreast. Permanecem abertos
releases finais, rótulos, autorização/acesso,
orçamento, algoritmo HPO, ranges de augmentation, agregação por paciente e
multiplicidade. Este documento não os aprova.
