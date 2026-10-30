# Pedidos de seeds por arquivo

Edite **control/seed-requests.md no branch control**, um bloco JSON por pedido.
BOARD.md é derivado. Preserve pedidos antigos e use request_id novo.

```json
{
  "request_id": "r-0001",
  "dataset": "isic2020",
  "campaign_sha256": "<SHA-256 completo da campanha selada e ancorada>",
  "seeds": [13, 19],
  "node_id": "<id do nó publicado em fleet/nodes>",
  "requested_on": "2026-10-30",
  "requested_by": "Eduardo",
  "decided_after_results": false,
  "cross_environment": false,
  "note": "Há tempo e orçamento para todos os braços e folds."
}
```

O schema é fechado; note e cross_environment são opcionais. cross_environment
aceita somente booleano e equivale a false quando ausente. O parser rejeita
chaves/blocos/IDs duplicados, seeds repetidas, seed 11, campos desconhecidos e
cercas abertas. Erro de schema invalida o arquivo inteiro.

## Regras científicas e de nó

- O adaptador real exige exatamente as seeds de um estágio posterior já
  declarado na campanha, na ordem selada. Não expande a política da campanha.
- A combinação campanha + seed fica vinculada a um pedido e nó. Seeds já
  existentes impedem outro pedido. São sempre dez braços × cinco folds por seed.
- O nó deve estar active, ter hardware_class, dataset/versão/manifest validados
  e uma imagem com images[].probe. A imagem é obrigatoriamente a da campanha.
- O environment_sha256 é o SHA-256 canônico de {image_id, probe}. Sem declaração,
  deve ser idêntico ao da campanha. Com **"cross_environment": true**, o environment
  do nó fica selado na autorização e os relatórios all_authorized marcam cada
  seed correspondente **different environment (declared)**. O runtime verifica
  esse environment em claim/início/retomada. A imagem continua fixa.
- decided_after_results deve ser false: decisão por tempo/orçamento. SAP v1,
  item 5, e análise primária (seed 11) permanecem iguais. Seeds posteriores são
  sensibilidade; o relatório padrão stage1 conserva seus bytes.
- Pedido, perfil, environment e stop_at são imutáveis após iniciar autorização.
  Alteração/remoção aparece como invalid. Não apague o estado para reaplicar.
- Prazo padrão: **2026-11-12T00:00:00Z**, inclusive. Nenhum início ou retomada
  após o prazo; jobs em execução podem terminar. --stop-at fixa outro prazo.

## Preparação do código e configuração do adaptador real

C9b exige um execution_code_commit existente: descendente do commit da campanha,
com todos os arquivos anteriores intactos, acrescentando somente os configs
exatos em configs/global/<dataset>/final/stage-N/ e seu MANIFEST.json.
O operador prepara esses arquivos com materialize-final --stage N e faz o commit
antes de aplicar o pedido. O adaptador nunca cria commits de código.
Veja [global-cli.md](global-cli.md) para a preparação C9b.

Crie um JSON privado de configuração (caminhos code_repo/data_root/source
relativos ao arquivo de configuração ou absolutos):

```json
{
  "code_repo": "/srv/code",
  "data_root": "/srv/private/data",
  "source": "/srv/private/source.json",
  "remote": "/srv/queue.git",
  "branch": "queue",
  "min_vram_mib": 1000
}
```

min_vram_mib é uma escolha explícita do operador para o experimento; o valor
acima é ilustrativo. source é o mesmo documento dataset source da CLI global.
O armazenamento privado deve conter pools, campanha, freeze e evidência HPO
quando aplicável. O HEAD de code_repo deve estar limpo no commit aditivo.

```bash
python -m augment_atlas.seed_requests plan \
  --control-checkout /srv/control --queue-checkout /srv/queue \
  --node-cards /srv/fleet/nodes --state-dir /srv/private/seed-request-state \
  --authorizer-config /srv/private/authorizer.json

python -m augment_atlas.seed_requests apply \
  --control-checkout /srv/control --queue-checkout /srv/queue \
  --node-cards /srv/fleet/nodes --state-dir /srv/private/seed-request-state \
  --authorizer-config /srv/private/authorizer.json
```

Com o adaptador, plan lê um único HEAD publicado da fila e audita os inputs
privados; não publica âncoras/jobs nem cria estado do pedido. A consulta Git
pode inicializar o cache local/fazer fetch. apply usa o relógio real; --now é
apenas para plan. Sem configuração, plan mantém a leitura offline dos arquivos
locais e apply exige um adaptador. --authorizer módulo:factory continua sendo
a interface para adaptadores instalados; não combine as duas opções.

## Transação e retomada

RealStageAuthorizer chama authorize_stage, audita/materializa o suplemento
aditivo, publica stage_authorization e devolve o recibo durável. O controlador
então acrescenta requirements.allowed_nodes=[node_id], hardware_class,
environment_sha256 e seed_request (hashes, âncora e prazo), e chama seed_final
com a autorização auditada. A âncora é ancestral estrito do commit dos jobs.
A publicação usa GitQueue: não depende de jobs locais ainda não publicados.

seed-requests-state.json registra a reserva antes da autorização e os specs
antes da publicação. O adaptador também guarda a autorização privada por hash
do pedido **antes** da primeira mutação na fila. Resposta perdida de âncora ou
seed-final reutiliza a mesma autorização/specs após reinício. Não migre uma
publicação real para o modo de arquivos ao retomar.

Use um state-dir persistente por fila. Locks exclusivos na fila e estado evitam
aplicações locais concorrentes. Após processo morto, inspecione-o antes de
remover seu lock. Jobs/desfechos existentes não são recriados nem apagados.

## Quadro e saídas

```bash
python -m augment_atlas.queue_board \
  --queue-checkout /srv/queue --control-checkout /srv/control \
  --node-cards /srv/fleet/nodes --state-dir /srv/private/seed-request-state \
  --out /srv/public/BOARD.md
```

O quadro usa o snapshot comprometido queue (origin/queue, senão queue local),
com pedido/estado locais. Informa pending, authorized, running, done, expired
ou invalid, erros/ações e contadores por seed. done exige 50 sucessos por seed;
falhas/interrupções continuam visíveis. expired é visão de controle, sem
reescrever o histórico da fila. Atualize a referência queue do checkout do
quadro pelo fluxo já usado pelo controlador.

Exit: 0 sem invalid (inclui espera/expired), 1 para erro/pedido inválido,
2 para argumentos inválidos. Validação: tests/test_seed_requests.py e
tests/test_seed_stage_authorizer.py; este último usa Git local descartável e
não executa Docker, treinamento ou rede externa.
