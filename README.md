# ED01_IC
Exercício 03 do Estudo Dirigido 01 de Inteligência Computacional

Considerações:
- Base Leaf: Gaussian Naive Bayes possui melhor score;
- Base Car: Bernoulli Naive Bayes possui melhor score, mesmo não sendo uma base binária;
- Base Congress: Complement Naive Bayes possui score melhor, porém instável(dependendo do split treino-teste). Os 3 classificadores mencionados são similares nesta base;
- Com exceção da base Leaf a árvore de decisão possui melhor score em relação aos classificadores naive Bayes da lib sklearn.
