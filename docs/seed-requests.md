# Pedidos de seeds por arquivo

Eduardo edita **control/seed-requests.md no branch control** do repositório de
controle no GitHub. O **BOARD.md é derivado** e não recebe pedidos.

Acrescente um bloco JSON por pedido. Use um request_id novo e mantenha os
pedidos anteriores no arquivo. Troque o hash pelo SHA-256 completo da campanha
já selada e ancorada, e node_id pelo identificador do cartão publicado em
fleet/nodes/. O exemplo abaixo pode ser copiado; os dois campos entre sinais
de menor/maior precisam ser substituídos.

```json
{
  "request_id": "r-0001",
  "dataset": "isic2020",
  "campaign_sha256": "<64 caracteres hexadecimais da campanha>",
  "seeds": [23],
  "node_id": "<id do nó>",
  "requested_on": "2026-10-30",
  "requested_by": "Eduardo",
  "decided_after_results": false,
  "note": "Há tempo e orçamento para completar todos os braços e folds."
}
```

O arquivo real usa cercas de três crases, com json imediatamente após a cerca
de abertura, como no exemplo. Texto explicativo fora dos blocos é permitido.
O schema é fechado: somente os campos acima; note é opcional. Datas usam
AAAA-MM-DD; seeds são inteiros não negativos, sem repetição. O parser rejeita
campos desconhecidos, chaves JSON repetidas, request_id repetido, JSON fora
das cercas e blocos sem fechamento. Um erro de sintaxe/schema torna o arquivo
invalid e impede qualquer aplicação parcial; o quadro informa a linha do erro.

## Regras do pedido

- A combinação campanha + seed pertence a um único nó, permanentemente.
  Outro pedido para a mesma seed é inválido, mesmo escolhendo outro servidor.
  Jobs existentes dessa seed também impedem um novo pedido. A seed 11 não é nova.
- O nó precisa existir, estar active, ter hardware_class e dataset/versão
  validados. Sua imagem precisa ter uma sonda em images[].probe.
- decided_after_results é obrigatório e deve ser false. A decisão deve vir de
  tempo/orçamento. SAP v1 selado, item 5: seeds adicionais são sensibilidade,
  somente com os dez braços × cinco folds completos. O pedido não altera o SAP
  nem a seed da análise primária.
- Depois que a autorização começa, o conteúdo inteiro do pedido fica preso ao
  hash, inclusive note, responsável e data. Alterar ou remover o bloco produz
  invalid visível. Não renomeie pedidos para contornar uma autorização anterior.
- A imagem permanece igual à da campanha; o ambiente vem da sonda dessa imagem
  no nó escolhido: SHA-256 canônico de {image_id, probe}. O perfil também fica
  registrado pelo profile_sha256. Mudanças posteriores exigem investigação;
  não apague o estado para forçar uma reaplicação.
- O limite padrão para iniciar jobs é **2026-11-12T00:00:00Z**, inclusive:
  a partir desse instante, nenhum novo início ou retomada é permitido.
  Jobs já rodando podem terminar. Pedidos com requested_on nesse dia ou depois
  ficam expired; datas futuras anteriores ao limite aguardam o dia solicitado.

## Planejar e aplicar no controlador

Use checkouts locais separados, atualizados pelo operador, do branch control
e do branch queue. O módulo lê os arquivos desses checkouts, inclusive alterações
locais; não faz fetch, commit, checkout de branch nem push. Use um único
state-dir persistente por fila e preserve esse diretório entre reinícios.

```bash
python -m augment_atlas.seed_requests plan \
  --control-checkout /srv/control \
  --queue-checkout /srv/queue \
  --node-cards /srv/fleet/nodes \
  --state-dir /srv/private/seed-request-state
```

A saída JSON lista pedido, ações, status, ambiente e quantidade esperada de jobs
(50 por seed). plan não cria arquivos. Para uma simulação histórica de plan,
há --now 2026-10-30T12:00:00Z. apply sempre usa o relógio real.

```bash
python -m augment_atlas.seed_requests apply \
  --control-checkout /srv/control \
  --queue-checkout /srv/queue \
  --node-cards /srv/fleet/nodes \
  --state-dir /srv/private/seed-request-state \
  --authorizer meu_adaptador:criar_authorizer
```

**O adaptador real ainda precisa ser ligado ao authorize-stage.** O nome acima
é ilustrativo: não é um módulo fornecido neste patch. Sem adaptador, apply
mostra invalid com erro StageAuthorizer e não cria jobs. Os testes usam fake.

Após a autorização, apply gera os arquivos jobs/<job_id>.md no checkout local
da fila. Todos levam requirements.allowed_nodes=[node_id], hardware_class do nó,
imagem fixada, environment_sha256 do nó e seed_request com hashes e prazo.
A publicação desses arquivos é uma etapa posterior do controlador. Ela deve
preservar a âncora autorizadora como ancestral estrito do commit dos jobs.
A criação local dos arquivos não significa publicação nem execução.

O prazo global pode ser configurado com --stop-at em UTC terminado em Z.
Use o mesmo valor em plan, apply e no quadro. O prazo é fixado nos jobs e no
estado ao autorizar; mudar esse valor depois é erro, assim como editar o pedido.

O arquivo seed-requests-state.json guarda pedido, hash, ambiente, recibo da
âncora, specs autorizados e seus hashes. A aplicação grava um diário antes da
chamada externa e antes de criar jobs, usa locks exclusivos no checkout da fila e no state-dir e mantém o progresso
de jobs existentes. Se houver interrupção após autorização ou criação parcial,
reexecute com os mesmos diretórios: os jobs faltantes serão gerados sem refazer
a autorização já confirmada. Depois de resposta perdida, o adaptador recebe
novamente o mesmo request_sha256 e precisa devolver a mesma autorização.
Um lock deixado por processo morto exige verificar o processo anterior antes
de remover o arquivo; o sistema não assume que um lock é obsoleto.

Códigos de saída: 0 para plano/aplicação sem invalid (inclui espera e expired);
1 quando existe pedido inválido/erro operacional; 2 para argumentos inválidos.

## Ver o quadro

```bash
python -m augment_atlas.queue_board \
  --queue-checkout /srv/queue \
  --control-checkout /srv/control \
  --node-cards /srv/fleet/nodes \
  --state-dir /srv/private/seed-request-state \
  --out /srv/public/BOARD.md
```

O quadro continua lendo jobs/âncoras de um commit queue fixado
(origin/queue, senão queue local), sem consultar jobs não publicados.
A seção **Seed requests** combina esse snapshot com os pedidos e estado locais.
Ela informa dataset, seeds, nó, erro/ação e progresso por seed.

| Status | Significado |
| --- | --- |
| pending | Pedido válido aguardando autorização ou a data solicitada. |
| authorized | Autorização confirmada; jobs podem estar aguardando publicação/execução. |
| running | Há jobs claimed ou running dessa seed; o restante mantém seus contadores. |
| done | Todos os 50 jobs por seed terminaram com sucesso. |
| expired | Prazo atingido, sem job ativo; jobs pendentes aparecem também como expired na contagem. |
| invalid | Erro de arquivo, campanha, nó, duplicação, alteração ou estado; veja Detail. |

O estado remoto da fila continua usando pending/claimed/running/etc.
expired é uma visão de controle, sem reescrever histórico de execução.
Falhas e interrupções permanecem nos contadores por seed e não viram done.
Se o estado já diz applied e o snapshot remoto ainda não contém todos os jobs,
o quadro mostra authorized com awaiting_queue_publication até a publicação
ficar visível. O apply, que lê os arquivos locais, acusa invalid se jobs de uma
aplicação concluída desaparecerem; não recrie histórico apagado sem investigar.

O publicador de quadro existente pode continuar sem os parâmetros de controle;
nesse caso a seção informa que nenhum pedido foi fornecido. Para preencher a
seção, a invocação do renderer no controlador precisa receber os três parâmetros.

## Interface de integração com authorize-stage

Em src/augment_atlas/seed_requests.py, StageAuthorizer é um Protocol Python:

```python
def authorize(
    self, *, request, request_sha256, node_card, environment, stop_at,
    queue_checkout
) -> AuthorizedStage:
    ...
```

O adaptador deve:

1. Revalidar a campanha selada, novidade das seeds e afinidade global, preservando
   imagem, freeze, pools, plano e SAP. Validar execution_code_commit pela política
   somente aditiva em configs/global. O nó fornece probe e environment_sha256,
   que podem diferir do ambiente do servidor da primeira seed.
2. Materializar/auditar os configs reais e retornar specs completos com config,
   config_sha256, requirements, commit de execução e vínculos científicos.
   Não basta trocar train_seed num spec antigo: isso deixaria o config errado.
3. Selar e publicar a stage_authorization, de forma idempotente pelo hash do
   pedido. Só retornar depois de a âncora estar durável; uma resposta perdida
   não pode gerar outra autorização.
4. Retornar AuthorizedStage(stage_sha256, anchor_commit, image_id, job_specs),
   contendo exatamente dez braços × cinco folds por seed. Não criar jobs.
5. Na integração real, ligar a publicação dos novos jobs e os gates de runtime,
   coleta e cronologia ao authorize-stage. Os gates atuais de campanha selada
   ainda precisam reconhecer as seeds/commits adicionais; este patch não os
   relaxa nem implementa autorização científica.

O módulo confere matriz, dataset, freeze, pools, imagem e hashes dos specs
recebidos, fixa afinidade e ambiente e escreve os jobs somente após o recibo.
A prova de que a âncora está publicada e de que o commit/config são autorizados
é responsabilidade do adaptador real. Os testes sintéticos não provam isso.
