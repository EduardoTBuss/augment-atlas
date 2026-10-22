# Contexto do trabalho na nuvem — estado em 22/10/2026

Este arquivo existe para que qualquer IA (local ou na nuvem) saiba o que já foi feito, o que
está pendente e qual é o próximo passo, sem reler toda a conversa. Leia junto com
[`cloud-handoff.md`](cloud-handoff.md) (regras duras) e [`global-cli.md`](global-cli.md)
(runbook do operador). Repositório público: aqui só há código, decisões e contagens de
testes. Nenhum dado privado, caminho, host, credencial ou resultado de campanha.

Atualize este arquivo ao fim de cada sessão de trabalho, mudando a data do título.

---

## 1. Resumo em uma tela

- **O código do protocolo D/E está completo para a campanha (a) do INbreast**, com a
  receita externa fixa. Isso vai de pools a freeze, âncora, configs, campanha, jobs finais,
  coleta, relatório e exportação para o artigo, tudo via `python -m augment_atlas.global_cli`.
- **A campanha (b), com HPO, está bloqueada por uma decisão de protocolo** (B1, §5).
- **O ISIC está bloqueado por uma decisão sobre seeds** e pelo checklist §7.1 do plano de
  aquisição.
- **Nada foi treinado nem executado em servidor.** `scientific_status` continua
  `"provisional"`, e `execution_verified` e `scientific_eligible` continuam `false`.
- **Nada foi mergeado em `main`.** O trabalho está em dois PRs empilhados, ambos em draft (§2).

## 2. Mapa de branches e PRs

| PR | Branch | Base | Conteúdo | Estado |
|---|---|---|---|---|
| EduardoTBuss/augment-atlas#3 | `claude/tarefa-c-merge-branches-pwmlw1` | `main` | Integração (tarefa C): as 5 branches de feature numa árvore só | draft, sem conflito |
| EduardoTBuss/augment-atlas#4 | `claude/magical-bardeen-vkyubk` | branch do #3 | Correção do sentinel, CLI global, coletor final, gerador de pools, este arquivo | draft, sem conflito |
| EduardoTBuss/augment-atlas#1, #2 | `wip/s4-telemetry-20261021`, `feature/global-protocol-20261016` | `main` | Já contidos no #3 | abertos |

Ordem de merge: **#3, depois #4**. Depois do merge do #3, os PRs #1 e #2 podem ser
fechados, porque o conteúdo deles já está no #3.

Branches de origem integradas no #3 (merge `--no-ff`, nesta ordem):

| Branch | Commit |
|---|---|
| `feature/global-protocol-20261016` (S0–S3 + S2) | `12b4ee3` |
| `wip/s4-telemetry-20261021` (S4 + telemetria de treino) | `2960a5e` |
| `fix/cpu-isa-avx2-20261020` (limite AVX2) | `d812ba3` |
| `feature/fleet-20261019` | `cff5d81` |
| `feature/isic-acquisition-20261019` | `e956b5e` |

## 3. O que foi feito (cronológico)

### 3.1 Integração (PR #3)
- **Conflitos resolvidos sem mudar comportamento:**
  - `runtime.py` `DockerRuntime.create`: a checagem de ISA continua primeiro e falha fechado,
    e a montagem da D-view global foi mantida.
  - `cli.py`: `recipe_identity` passou a ser importado de `contracts`.
  - `queue.py`: ficaram as âncoras S3, `seed_final` e os documentos do fleet.
  - `docs/global-next-integration.md`: os blocos de erro do S2 e do S4 foram concatenados.
- **Revisão:**
  - todos os arquivos de `tests/` são idênticos byte a byte aos da branch de origem;
  - nenhuma flag científica é `true`;
  - os 8 casos adversariais do S2 têm teste;
  - uma revisão independente confirmou as resoluções.

### 3.2 PR #4

| Commit | O quê |
|---|---|
| `c4aefd2` | **Correção F2 do sentinel.** Uma ferramenta ausente do PATH virava `permission_denied` quando o PATH tinha um diretório ilegível, como um usuário sem privilégios herdando o PATH do root. Agora sai `tool_absent`. |
| `4f8831e` | **`global_cli.py`, `global_collect.py` e `docs/global-cli.md`.** A CLI do operador (spec §13) e o coletor de jobs finais que o `build_report` exige (spec §6.10). |
| `8425f6f` | **Correções da 1ª revisão independente:** `results.json` checksummado, modo de validação explícito, varredura de IDs/grupos/caminhos também no stderr, relatório preso aos pools e uma ferramenta não executável continua `permission_denied`. |
| `e6aa7f3` | **`global_splits.py` e `make-pools`.** Gerador de pools D/E `global-dev-eval-v1` (§5 do `dataset-acquisition-plan.md`), só stdlib. |
| `7510b38` | **Correções da 2ª revisão independente:** frações e limiares presos ao config pré-registrado, arredondamento exato, telemetria inválida vira `null` e não derruba o relatório. |
| (este) | Este arquivo de contexto e a atualização do `cloud-handoff.md`. |

**Módulos novos:**

| Módulo | Função |
|---|---|
| `global_cli.py` | Subcomandos `identity`, `make-pools`, `import-pools`, `audit-pools`, `hpo-view`, `d-view`, `seal-external`, `materialize-final`, `seal-campaign`, `anchor`, `seed-final`, `audit-chronology`, `report`, `summary`, `export-article`, `compare`. Lê documentos privados por hash em `<data-root>/global/<ds>/<kind>-<sha>.json`, reaudita tudo e grava sem sobrescrever. |
| `global_collect.py` | Monta as entradas do relatório a partir da fila e dos diretórios de attempt, sempre recalculando. |
| `global_splits.py` | Gera os pools D/E estratificados por grupo positivo × quartil de tamanho do grupo. Não troca de seed nem tem fallback. |

**Testes novos** (nenhum teste antigo foi alterado): `tests/test_sentinel_tool_resolution.py`,
`tests/test_global_cli.py` (ponta a ponta com arquivos reais, repositório de código real e
fila Git bare) e `tests/test_global_splits.py`.

## 4. Testes (último estado: commit `7510b38`)

Comando: `PYTHONPATH=src python3 -m unittest discover -s tests`. Python 3.13, sem torch.

| Execução | Rodados | Passaram | Pulados | Falharam |
|---|---|---|---|---|
| root | 900 | 864 | 35 | 1 (F1) |
| usuário sem privilégios (`nobody`), com o PATH do root | 900 | 866 | 34 | 0 |

- **Pulados:** são os testes de torch/contêiner, mais (só como root) o teste de exec real de
  EACCES. Pulado não é aprovado.
- **F1:** `test_sentinel.CliAndScriptTests.test_cli_run_report_latest` falha como root
  porque o sentinel recusa root de propósito. Não há correção que respeite as regras, então
  **rode a suíte com um usuário sem privilégios**.

## 5. Decisões pendentes do Eduardo (bloqueiam trabalho)

| # | Decisão | Opções | Bloqueia |
|---|---|---|---|
| **B1** | Commit dos jobs HPO. Os configs HPO embutem `plan_sha256`, que cobre `plan.runtime.commit`. O coletor recalcula os ids dos jobs com esse commit (`code_commit` é campo de identidade), e o worker exige o config commitado nesse mesmo commit. Um arquivo não pode conter o hash do commit em que ele está. | (i) Regra tipo Q6 para HPO: o commit dos configs HPO pode diferir de `plan.runtime.commit` só em `configs/global/**`, com o mesmo código. (ii) Tirar o commit do hash embutido nos configs. | Campanha (b) do INbreast e qualquer HPO; os subcomandos HPO da CLI |
| **S1** | Seed de avaliação do ISIC. O config `configs/acquisition/isic2020-v1.json` tem uma `split_seed` só, mas o schema de pools e a Q3 exigem uma seed de avaliação distinta. | Emenda ao config declarando `development`, `selection` e `evaluation` (a de avaliação diferente das outras duas). | `make-pools` do ISIC |
| **S2** | Detalhes que a §5 não fixa. Implementados assim, confirmar ou mudar: quartis nearest-rank sobre todos os grupos, com o D_select reaproveitando os estratos de D; arredondamento meio-para-cima exato (o `round` do Python dá 4 em vez de 5 com f_sel = 0,30 num estrato de 15 grupos); rodízio único dos folds entre estratos. | Confirmar ou pedir mudança | Splits do ISIC/mBRSET |
| **S3** | Estratificação de tarefa ordinal/multiclasse (mBRSET, QWK). Não especificada. | Definir a regra | `make-pools` do mBRSET |
| **Q6** | `same_code_identity_v1` continua `pending_eduardo_confirmation` no código. | Confirmar | Selo de campanha "definitivo" |
| **ISIC §7.1** | Licença, os 4 links, emenda de tarefa, resolução 512, política de duplicatas, frações e autorização do download. | — | Aquisição do ISIC |

## 6. Achados que ficaram registrados (sem correção)

- **Arquivo central, fora do escopo:** o worker (`runtime.py`, `_global_config_bytes`) lê
  config e MANIFEST da árvore de trabalho, não do blob do HEAD. O conteúdo continua preso
  por hash, então não é falha aberta. A correção sugerida é ler com
  `git cat-file blob HEAD:<path>`.
- **Seed única, mais de uma leitura:** a §5 menciona uma `split_seed` única; o gerador usa
  três seeds (ver S1).

## 7. Próximos passos (em ordem)

**Para o Eduardo:**
1. Revisar e mergear o #3 e depois o #4. Fechar os PRs #1 e #2.
2. Decidir B1, S1 e S2 (e Q6 se quiser fechar). Mandar as decisões para a sessão na nuvem.

**No servidor (só o Eduardo; a nuvem não acessa servidor):**
3. Piloto INbreast, campanha (a), seguindo [`global-cli.md`](global-cli.md):
   1. `identity` (conferir as 387 imagens);
   2. `import-pools`, `audit-pools`;
   3. calibrar o throughput (1 job smoke com dados de D ou sintéticos);
   4. `seal-external` e `anchor recipe_freeze`;
   5. `materialize-final`, commit, `seal-campaign` e `anchor campaign`;
   6. `seed-final` e os workers `phase=final`;
   7. `report`, `summary` e `export-article`.

   Teto de 8 GPU-h. Se estourar, parar e reportar a matriz incompleta.
4. Rodar a suíte com um usuário sem privilégios no servidor antes do primeiro job real.

**Na nuvem (código, depois das decisões):**
5. B1 decidida: implementar a regra e os subcomandos HPO (`search-space`, `build-plan`,
   `materialize-hpo`, `seed-hpo`, `collect-hpo`, `seal-freeze`), com testes ponta a ponta.
6. S1/S2 decididas: ajustar o config do ISIC (emenda) e o gerador. S3 decidida:
   estratificação ordinal para o mBRSET.
7. Opcional: ler os configs do worker a partir do blob do HEAD (§6), num PR separado,
   porque mexe em arquivo central.
8. Depois do piloto aceito: S6 multilabel (CheXpert), que está fora de escopo por enquanto.

## 8. Como a sessão na nuvem trabalha (para manter o padrão)

- Laço do handoff: implementar, rodar testes, revisão independente em contexto novo e
  somente leitura, corrigir os achados, repetir.
- Testes antigos nunca são editados. Todo caso adversarial tem teste, e mutações são
  reassinadas para o checksum não esconder a falha.
- Lógica nova vai em módulo novo. Os arquivos centrais (`train/runtime/contracts/evaluation/cli`)
  só recebem ganchos pequenos.
- Commits sem trailers de co-autoria. Nada vai direto para `main`. PRs ficam em draft até o
  Eduardo revisar.
