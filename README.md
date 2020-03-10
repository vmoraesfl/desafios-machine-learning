# Finalização do Desafio Cyberlabs
### **Beat Human Performance!** 🌟

---
Este desafio se mostrou muito importante para que eu absorvesse conhecimentos relevantes de data science e machine learning. Nele, pude conhecer ferramentas novas e ganhar maturidade no assunto.

Meu primeiro passo foi olhar o dataset, como autorizado por vocês, para conseguir os nomes das colunas com indicadores biológicos. Fiz isso para tentar identificar e eliminar alguma possível coluna que causasse data leakage, como por exemplo uma coluna post factum. 

Ao pensar sobre qual seria a melhor maneira de resolver o desafio, decidi que eu deveria ir atrás de modelos de classificação variados, aprender sobre eles, implementá-los no dataset e analisar os resultados de cada um.

Para isso, eu precisaria escolher qual parâmetro eu usaria para avaliar o sucesso do modelo implementado. Decidi usar o cross validation e o F1 score.

O primeiro algoritmo utilizado foi o Random Forest. No início testei na mão vários números de árvores diferentes pq ainda não conhecia o método do Grid Search. 

Para continuar implementando os outros algoritmos, usei um template que havia visto num curso da Udemy afim de otimizar a organização do código. Em seguida implementei os seguintes modelos: Logistic Regression, Kernel SVM e XGBoost.

Utilizei o Grid Search para otimizar os parâmetros e testei os resultados usando a seed 42. Os testes mostraram que o Kernel SVM apresentou resultados um pouco melhores que os demais e, por isso, foi o escolhido para ser enviado como resposta do desafio. 

 🤘
