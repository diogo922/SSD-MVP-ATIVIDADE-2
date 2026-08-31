# Sistema de Suporte à Decisão para Implantação de Nova Unidade de Armazenamento em Nuvem

**Análise descritiva, preditiva e prescritiva de risco, custo e prazo**
Disciplina: Sistemas de Suporte à Decisão

---

## 1. Contextualização do problema

Uma empresa de armazenamento de dados em nuvem precisa abrir uma nova unidade (data center) com
**10 PB de capacidade útil**. A decisão envolve três dimensões que normalmente são tratadas
separadamente — e é justamente aí que os projetos falham:

| Dimensão | Pergunta de decisão |
|---|---|
| **Risco** | Quantos discos vão falhar? Qual a exposição a perda de dados? |
| **Custo** | Quanto custa em 5 anos, considerando falhas e operação — não só a compra? |
| **Prazo** | Em quanto tempo a unidade entra em operação plena, e com que confiança? |

O problema real não é escolher o disco mais barato. É que **a decisão de compra determina o
custo operacional dos cinco anos seguintes**, e essa consequência não aparece na cotação do
fornecedor. Um disco 10% mais barato com o dobro da taxa de falha é um passivo, não uma economia.

O objetivo deste MVP é construir um **motor de decisão auditável** que conecte evidência empírica
de confiabilidade a uma recomendação de configuração, orçamento e cronograma.

### Base de dados

**Backblaze Hard Drive Data** (Kaggle) — telemetria SMART diária de discos em operação real.

| Característica | Valor |
|---|---|
| Observações | 3.179.295 registros (disco-dia) |
| Discos únicos | 65.993 |
| Modelos | 69 (11 com exposição suficiente para análise) |
| Falhas registradas | 215 |
| Período | 50 dias observados (jan e abr/2016) |
| Atributos | 90 métricas SMART + metadados |



## 2. Auditoria de qualidade dos dados

### 2.1 Coluna `capacity_bytes` 

O valor armazenado é `1.97665142785814e-311` — um número *denormal*, praticamente zero.

**Diagnóstico:** o valor original era um inteiro de 64 bits (4.000.787.030.016 bytes = 4 TB),
gravado no CSV com os **bits reinterpretados como `float64`**. Os bits estão íntegros; o tipo
declarado é que está errado.

**Correção:** `df['capacity_bytes'].to_numpy().view('int64')` reinterpreta a mesma região de
memória com o tipo correto. Todas as 12 capacidades (0,08 TB a 8 TB) foram recuperadas.

Sem essa correção, qualquer análise por capacidade — inclusive o dimensionamento da frota —
seria impossível.

### 2.2 Descontinuidade temporal não documentada

O período aparenta cobrir jan–abr/2016, mas contém apenas **50 dias**: 01–21/jan e 01–29/abr.
Fevereiro e março estão ausentes.

**Consequência:** análise de sobrevivência em tempo calendário, séries temporais e cálculos de
"dias até a falha" produziriam resultados enviesados.

**Mitigação:** toda a análise usa **taxa por disco-dia**, não tempo calendário. Como taxa é uma
razão entre eventos e exposição observada, lacunas no calendário não a enviesam.

---

## 3. Hipóteses

| # | Hipótese | Teste | Resultado |
|---|---|---|---|
| **H1** | A taxa de falha difere significativamente entre modelos de mesma capacidade | AFR com IC exato de Poisson; verificação de sobreposição dos intervalos | **Confirmada** |
| **H2** | Atributos SMART contêm sinal detectável antes da falha | Comparação de distribuições e classificador supervisionado | **Confirmada** |
| **H3** | Manutenção preditiva reduz o custo total de propriedade | Simulação de TCO, política reativa vs. preditiva | **Confirmada, com ressalva relevante** |
| **H4** | A escolha do modelo de disco impacta o TCO mais que a política de manutenção | Comparação das magnitudes de economia | **Confirmada** |
| **H5** | O prazo de implantação tem variância suficiente para invalidar cronograma de data única | Simulação PERT + Monte Carlo com risco correlacionado | **Confirmada** |

---

## 4. Análises e resultados

### 4.1 Descritiva — o risco não está distribuído por igual

AFR (*Annualized Failure Rate*) calculado como `falhas ÷ disco-dias × 365`, com **intervalo de
confiança exato de Poisson**. O IC não é decoração: com eventos raros, uma estimativa pontual sem
intervalo é um convite à decisão errada.

| Modelo | Cap. | AFR | IC 95% | Disco-dias |
|---|---|---|---|---|
| HGST HMS5C4040ALE640 | 4 TB | **0,20%** | 0,02 – 0,72% | 368.077 |
| HGST HMS5C4040BLE640 | 4 TB | 0,20% | 0,00 – 1,09% | 186.832 |
| ST6000DX000 | 6 TB | 0,75% | 0,09 – 2,69% | 97.864 |
| Hitachi HDS5C3030ALA630 | 3 TB | 0,77% | 0,25 – 1,80% | 236.690 |
| Hitachi HDS5C4040ALE630 | 4 TB | 1,07% | 0,29 – 2,73% | 136.969 |
| Hitachi HDS722020ALA330 | 2 TB | 2,12% | 1,13 – 3,62% | 224.052 |
| **ST4000DM000** | 4 TB | **3,02%** | 2,54 – 3,56% | 1.681.473 |
| WDC WD30EFRX | 3 TB | 4,00% | 1,47 – 8,72% | 54.686 |

**Achado central:** dois discos de **capacidade idêntica (4 TB)** apresentam AFR de **0,20%** e
**3,02%** — diferença de **~15x**, com intervalos de confiança que **não se sobrepõem**. Não é
ruído amostral.

Em termos operacionais, numa frota de 10.000 discos isso é a diferença entre ~20 e ~300
substituições por ano. A escolha do modelo é decisão estratégica, não item de compras.

### 4.2 Preditiva — o sinal existe, mas tem limite

**Desenho experimental.** O dataset é disco-dia; treinar diretamente nas 3,18 M linhas com split
aleatório colocaria o mesmo disco em treino e teste — **vazamento de dados**. Os dados foram
colapsados para **um registro por disco** (última observação), com split estratificado.

**Modelo.** `HistGradientBoostingClassifier` — escolhido por tratar `NaN` nativamente
(`smart_187` tem 43% de ausência estrutural: HGST/Hitachi não reportam o atributo), capturar
interações não-lineares e suportar rebalanceamento de classe.

| Métrica | Valor |
|---|---|
| ROC-AUC | 0,78 |
| PR-AUC | 0,311 |
| Baseline aleatório | 0,0031 |
| **Lift** | **~99x** |
| Revocação no limiar ótimo | 31% |
| Falsos positivos | 0,10% da frota |

**Por que PR-AUC e não acurácia.** Com prevalência de 0,31%, um modelo que responde "nada vai
falhar" acerta 99,7%. Acurácia é métrica inútil aqui. PR-AUC mede o desempenho na classe rara,
que é a única que importa.

**Limitação declarada:** desempenho nessa faixa **não autoriza substituição automática de disco**.
A taxa de falsos positivos é relevante demais. O que ele autoriza — e é o uso proposto — é
**priorização de inspeção**. O sistema entrega uma fila de prioridade, não uma sentença.

**Limiar por custo.** A conversão de probabilidade em ação é decisão econômica, não estatística.
Varrendo os limiares para minimizar `FN × R$ 4.200 + FP × R$ 700`, o ótimo cai em **0,925** —
não em 0,5. (O corte fica alto porque `class_weight='balanced'` infla as probabilidades; sem
rebalanceamento cairia bem abaixo.) A lição não é a direção do ajuste: é que **o 0,5 padrão do
`predict()` não é ótimo para nenhuma decisão real.**

**Previsão agregada.** Para dimensionar estoque e equipe usa-se um modelo **bayesiano
Gama-Poisson**, que separa duas incertezas distintas: (a) não conhecemos a taxa exata — posteriori
Gama com prior de Jeffreys; (b) mesmo conhecendo, o processo é aleatório — amostragem Poisson.
Uma simulação baseada só no AFR pontual ignora (a) e **subestima o risco de cauda**, que é
exatamente o que dimensiona estoque.

### 4.3 Prescritiva — a decisão

Motor Monte Carlo (10.000 cenários por configuração) que propaga a incerteza de falha até TCO,
estoque e equipe. Cada cenário é um futuro internamente coerente.

**Modelagem da manutenção preditiva:** ela não elimina falhas — **converte falha emergencial
(R$ 4.200) em troca planejada (R$ 700)**. A conversão usa a revocação medida (31%) e os falsos
positivos são cobrados como custo real. Sem esse rigor, o ganho do modelo apareceria inflado.

**TCO em 5 anos, 10 PB úteis:**

| Modelo | Discos | CAPEX | Falhas 5a (média / p95) | TCO médio | Exposição de cauda |
|---|---|---|---|---|---|
| **ST6000DX000** | 2.333 | R$ 4,42 MM | 109 / 239 | **R$ 6,06 MM** | R$ 0,41 MM |
| WDC WD60EFRX | 2.333 | R$ 4,42 MM | 268 / 690 | R$ 6,55 MM | R$ 1,33 MM |
| **HGST HMS5C4040ALE640** | 3.500 | R$ 5,18 MM | 44 / 98 | R$ 6,91 MM | **R$ 0,17 MM** |
| ST4000DM000 | 3.500 | R$ 5,18 MM | 530 / 614 | R$ 8,43 MM | R$ 0,27 MM |
| WDC WD30EFRX | 4.666 | R$ 5,90 MM | 1.005 / 1.732 | R$ 10,94 MM | R$ 2,28 MM |

**Fronteira eficiente.** Após eliminar configurações dominadas (mais caras *e* mais arriscadas
que alguma alternativa), restam **duas** opções racionais:

- **ST6000DX000** — menor custo esperado (R$ 6,06 MM), maior exposição de cauda (R$ 0,41 MM)
- **HGST HMS5C4040ALE640** — custo maior (R$ 6,91 MM), **exposição de cauda 2,4x menor** (R$ 0,17 MM)

A escolha entre as duas **não é estatística — é de apetite a risco**. O DSS delimita as opções
racionais e quantifica o trade-off; a diretoria decide dentro dele. Para operação sob SLA rígido
de durabilidade, os R$ 0,85 MM adicionais do HGST compram uma redução material de variância.

**Cronograma (PERT + Monte Carlo, 20.000 cenários).** Duas decisões de modelagem mudam o
resultado: (a) **compra e obra correm em paralelo** — somar em série infla o prazo em meses;
(b) **atrasos são correlacionados** (mesma equipe, mesmo fornecedor, mesmo cenário macro), então
um choque sistêmico multiplica todas as etapas. Sem isso, as variâncias se cancelam pelo Teorema
Central do Limite e o cronograma sai otimista demais.

| Percentil | Prazo | Uso |
|---|---|---|
| p50 | 208 dias (6,9 meses) | Meta interna |
| **p80** | **246 dias (8,2 meses)** | **Compromisso contratual** |
| p95 | 289 dias (9,6 meses) | Pior caso planejado |

Buffer p50→p80: **38 dias**. Prometer o p50 ao cliente é assumir ~50% de probabilidade de atraso.

**Análise de sensibilidade (tornado).**

| Premissa | Baixo | Base | Alto | Amplitude |
|---|---|---|---|---|
| Custo por TB | 5,18 | 6,06 | 6,94 | **R$ 1,76 MM** |
| Overhead de erasure coding | 5,31 | 6,05 | 6,84 | **R$ 1,53 MM** |
| Custo de troca emergencial | 5,90 | 6,06 | 6,38 | R$ 0,48 MM |
| OPEX de energia | 5,89 | 6,06 | 6,30 | R$ 0,41 MM |

O TCO é dominado por **preço de aquisição** e **esquema de redundância** — não por custo de
manutenção. Isso realoca o esforço gerencial: negociar compra e revisar o erasure coding valem
mais que otimizar o processo de troca.

---

## 5. Conclusões

**1. A escolha de hardware domina o projeto.**
Migrar do pior para o melhor candidato reduz o TCO em **~45%** (R$ 10,94 MM → R$ 6,06 MM). Nenhuma
otimização de processo posterior chega perto desse efeito. É a decisão de maior alavancagem, e é
tomada uma única vez, no início.

**2. Manutenção preditiva não compensa má escolha de hardware.**
A economia da política preditiva varia de **0,5%** (discos confiáveis) a **9,1%** (discos ruins).
Ou seja: o ML rende mais justamente onde o hardware é pior — o que significa que ele está
*compensando* um erro de compra, não gerando valor novo. A ordem correta de decisão é
**(1) escolher o disco certo, (2) depois otimizar a manutenção.** Inverter essa ordem é otimizar
o galho errado da árvore.

**3. Estimativa pontual não é decisão.**
Estoque dimensionado pela média cobre ~50% dos cenários. O sistema dimensiona por **p95**, e
provisiona reserva de contingência explícita (R$ 0,41 MM) em vez de tratar a cauda como surpresa.
O mesmo vale para o prazo: o compromisso externo é o p80, não o p50.

**4. Auditoria de dados é parte da análise, não etapa preliminar.**
Dois defeitos silenciosos — uma coluna com tipo corrompido e uma lacuna temporal não documentada —
teriam invalidado o dimensionamento e a análise de sobrevivência. Nenhum dos dois gera erro de
execução; ambos geram resultado errado com aparência de correto.

**5. O valor do DSS está na separação entre evidência e premissa.**
Risco foi medido; custo e prazo foram parametrizados e testados por sensibilidade. Isso permite
que qualquer premissa seja contestada e a decisão recalculada — o que um número único em slide
não permite.

---

## 6. Limitações

Declaradas de forma explícita, porque um DSS que omite suas limitações induz decisão errada com
aparência de rigor.

1. **Janela curta.** 50 dias observados, com lacuna de fevereiro–março. Não captura sazonalidade
   nem a curva de banheira completa (falha infantil e desgaste terminal).
2. **Dados de 2016.** O hardware analisado está obsoleto. A **metodologia transfere; os números
   específicos de AFR, não.** Devem ser recalculados com dados atuais antes de qualquer compra.
3. **215 falhas no total.** Modelos com poucos eventos têm IC largo. Daí o corte de exposição
   mínima (20.000 disco-dias) e o uso de IC em toda estimativa — o que limita, mas não elimina,
   o problema.
4. **Viés de seleção.** O ST4000DM000 representa ~53% dos disco-dias. A frota reflete a política
   de compras de um único operador, não um experimento controlado.
5. **Custo e prazo são premissas.** Limite estrutural do dataset. A análise de sensibilidade
   mitiga, mas não substitui cotação real de fornecedor.
6. **Ambiente único.** Um operador, um perfil de carga. Generalizar exige validação.

**Caminho para produção:** substituir o CSV por telemetria atual e o dicionário `P` por cotações
reais converte o protótipo em ferramenta operacional **sem reescrever a lógica de decisão**. Essa
separação foi um requisito de projeto, não um acidente.

---

## 7. Como executar

1. Baixe o dataset **Backblaze Hard Drive Data** no Kaggle (`archive.zip`, ~211 MB)
2. Abra `DSS_nova_unidade.ipynb` no Google Colab
3. Execute a **célula 0**, clique em **Escolher arquivos** e anexe o zip
4. Menu **Executar > Executar tudo**

**Requisitos:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib` — todos pré-instalados
no Colab. Tempo de execução: ~5 minutos. Semente fixa (`RNG = 42`): resultados reproduzíveis.

## 8. Estrutura do repositório

```
├── DSS_nova_unidade.ipynb    # Notebook completo, comentado célula a célula
├── README.md                 # Este documento
└── data/
    └── harddrive.csv         # Dataset (não versionado — ~1,2 GB)
```

## 9. Metodologia aplicada

| Técnica | Aplicação |
|---|---|
| IC exato de Poisson | Quantificação da incerteza do AFR |
| Gradient Boosting (histograma) | Classificação de falha com dados ausentes |
| PR-AUC / curva precisão-revocação | Avaliação sob desbalanceamento extremo |
| Otimização de limiar por custo | Conversão de probabilidade em ação |
| Inferência bayesiana Gama-Poisson | Previsão agregada com incerteza propagada |
| Simulação Monte Carlo | TCO, dimensionamento de estoque e equipe |
| Dominância de Pareto | Identificação da fronteira eficiente |
| PERT + choque correlacionado | Cronograma probabilístico |
| Análise de sensibilidade (tornado) | Robustez das premissas |
