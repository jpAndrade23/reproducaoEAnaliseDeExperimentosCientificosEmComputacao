# reproducaoEAnaliseDeExperimentosCientificosEmComputacao

<p>Este é o código de reprodução do experimento do artigo <a href='https://ieeexplore.ieee.org/document/9356267'>Performance of CatBoost and XGBoost in Medicare Fraud Detection</a></p>
<p>Utilizamos os algoritmos de ensemble testamos o AUC, tempo de construção das árvoes e tempo de execução do teste para 100, 250 e 500 árvores.</p>
<p>Aplicamos o teste utilizando os parâmtros para treinamento fornecidos no artigo, depth 6 e learn rate 0.1.</p>
<p>Como o objetivo era calcular o AUC era necessário um dataset de classificação binária, portanto optamos por usar o dataset <a href='https://archive.ics.uci.edu/dataset/2/adult'>Adult <a> que classifica se uma pessoa tem renda menor ou maior que 50k, para utilizar este dataset com o catbost e o xgboost foi necessário converter as colunas para valores numéricos.</p>
<p>As saídas da execução estão diferentes pois o dataset usado não é o mesmmo do experimento pois ele é indiponível para uso e também a uma grande diferença de hardware.</p>
<img src='https://media.discordapp.net/attachments/1082383095078076509/1180557552468107424/image.png?ex=657ddae6&is=656b65e6&hm=91efe26798e7348e5842c79cdf57c4b978c14559956e4e67e9f9294344077f59&=&format=webp&quality=lossless&width=1206&height=115' />
<p>Essas foram as saídas produzidas, optamos por não exibir a média como no experimento original pois não executamos o algoritmo muitas vezes como no artigo.</p>
<p>O desenvolvimento desta atividae foi uma grane experiência de aprendizado pois está é a primeira vez que aplicamos a teoria em código utilizando a biblioteca sklearn, xgboost e catboost, também nos permitiu experiênciar como o python é poderoso para tratamento de dados e como existem diferentes formas de preparar os dados para treinamemto sendo algumas melhores e outras piores.</p>
