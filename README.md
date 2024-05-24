
Machine Learning – Rapport



1. Fonctionnement du code: 
	
	L’exécution du script prend en paramètres optionnels les données à traiter wisc pour wisconsin.dat et dj pour dow_jones_index.data, pour les données à choix.

	Il prend aussi les méthode de classification PPV pour le plus proche voisin, AD pour les arbres de décision, PMC pour le perceptron multi-couches et finalement SVM pour la machine à vecteurs de support.

	Le script va donc calculer pour chaque ensemble de données et de classificateurs indiqués avec ses paramètres (définis en durs dans le programme pour des raisons de praticité) puis produire le fichier .csv correspondant.

	Ceci permet de comparer les résultats des différents modèles mais aussi quels paramètres influencent le mieux leur taux de classification (TCC).

Exemple d’utilisation :	>>> python3 dj wisc PMC AD
	
	Produit les 4 fichiers dans le dossier output : dj_pmc.csv, dj_ad.csv, wisc_pmc.csv, wisc_ad.csv contenant les résultats de wisconsin et du dow jones pour les méthodes du perceptron et des arbres de décision.


2. Données choisies

	J’ai choisi les données de la bourse dow_jones_index.data. Presque tous les attributs sont continus. Afin de travailler avec des classificateur, la classe à prédire à été définie à partir de l’attribut percent_return_next_dividend. Ainsi une valeur inférieure à 1 indique une perte et donc appartenant à la classe 0 si elle est supérieure à 1 ceci indique un gain et donc appartenant à la classe 1.
	Les données manquantes ont été comblées avec la valeur moyenne des exemples. Finalement tous les attributs continus ont été normalisés.


3. Mesure de performances

3.1 Temps de calcul

Les temps d’entraînement par modèle sont plutôt négligeables:

Évidemment, plus un estimateur possède de paramètres, plus il sera coûteux de trouver le meilleur ajustement, car cela implique de calculer un modèle supplémentaire pour chaque combinaison de paramètres. Il est donc judicieux de les limiter et de les choisir de manière intelligente.

Données de wisconsin.dat
Classificateur
Nombre de modèles entraînés
Temps médian pour l’entraînement d’un modèle [msec]
Temps total [secondes]
AD
32076
0
247
PPV
50
0
4,94
PMC
432
731
774

Données de dow_jones_index.data
Classificateur
Nombre de modèles entraînés
Temps médian pour l’entraînement d’un modèle [msec]
Temps total [secondes]
AD
32076
3,2
525
PPV
50
0,9
5,8
PMC
432
837
868


Nous pouvons observer que l’AD et le PPV sont peu coûteux en calculs (temps médian par modèle) contrairement au PMC  qui est assez lourd en comparaison.


3.2 Validation croisée

Le nombre de sous échantillons que l’on choisis pour la validation croisée augmente grandement les temps de calculs car nous multiplions le nombre de résultats par le nombre d’échantillons à entraîner. J’ai donc commencé avec une division en 3 sous-échantillons afin de trouver les paramètres et ai terminé avec 10 segments comme demandé pour obtenir des résultats plus réalistes en les moyennant.



3.3 Taux de classification

	Voici le meilleur taux de classification correct (TCC) moyen en calculant la moyenne des résultats obtenus de la validation croisée avec dix sous-échantillons (méthode : k-fold cross-validation).

Résultats pour wisconsin.dat
Classificateur
TCC moyen test 
TCC moyen entraînement
AD
0.88
0.88
PPV
0.97
0.98
PMC
0.97
0.98

Résultats pour dow_jownes_index.data
Classificateur
TCC moyen test 
TCC moyen entraînement
AD
0.92
0.94
PPV
0.88
1
PMC
0.97
0.98

Les performances entre classificateurs sont assez proches. 
Les TCC des sets d’entraînement sont légèrement supérieur aux TCC des tests ce qui est attendu car le modèle est entraîné avec le set d’entraînement par définition. Mais la différence de valeur entre TCC test et TCC entraînement n’est pas assez grande pour indiquer avec assurance un overfitting. 
Seule la valeur du TCC au-delà de 95 % indiquerai une performance trop grande et donc un overfitting sur nos données malgré l’utilisation de la validation croisée.


3.3.1 Plus Proche Voisin

Paramètres modifiés: 
    1. Nombre de vosins pour la requête: nombre impair de 1 à 50
    2. Poids pour la requête: uniform, distance

Ce classificateur ne possédant que deux paramètres nous intéressant dont un binaire nous pouvons représenter les résultats sur un graphe.


Datas : wisconsin.data


Nous avons un maximum à 5 et 7 voisins, ensuite l’augmentation de ce nombre n’améliore pas la performance. Les dents de scies sur le même nombre de voisins est causé par le mode de détermination de la classe (par distance ou par majorité). Avec ces données, le changement des paramètres influence peu la performance (différence de 1,8 point entre le maximum et le minimum).

Datas : dow_jownes_index.data 


Cette courbe fait penser à la fonction logarithme, le TCC s’améliore avec l’augmentation du nombre de voisins pour définir la classe.


3.3.2 Arbre de Décision

Paramètres modifiés : 
    1. Profondeur maximale (max_depth) : 1 à 99
    2. Nombre d’exemples minimum par feuille (min_samples_leaf) : 2 à 20
    3. Nombre minimal pour le découpage d’une node (min_samples_split) : 2 à 20
       
 Datas : wisconsin.data
mean_test_
score
std_test_score
mean_train_
score
mean_fit_
time
max_depth
min_samples_leaf
min_samples_split
0.875
0.03
0.88
0.0016
1
2
2
0.875
0.03
0.88
0.0015
99
19
19

Le découpage d’une node se fait avec le meilleur gini par défaut.
Ici nous pouvons observer que la profondeur maximale (max_depth)  n’influencent pas le TCC (meilleurs mean_test_score de 0.875) pour une certaine combinaison de min_simples_leaf et  min_samples_split. 

Il est tout a fait possible que certains paramètres comme le nombre d’exemples minimum dans une feuille (min_samples_leaf) arrêtent la croissance de l’arbre avant d’arriver à la profondeur maximale, ainsi max_depth ne produit aucune amélioration du TCC à partir d’un certain seuil. Nous pouvons avoir le même phénomène avec le nombre d’exemple minimal pour découper une node. Ces paramètres s’influence donc entre eux mais sont utiles pour se prémunir de l’overfitting, le pire des cas étant d’avoir une feuille pour chaque exemple.

Datas : dow_jownes_index.data

mean_test_
score
std_test_score
mean_train_
score
mean_fit_
time
max_depth
min_samples_leaf
min_samples_split
0.923
0.09
0.94
0.0018
3
2
7
0.923
0.09
0.94
0.0017
3
19
19

Sur ces données ce sont la profondeur de l’arbre qui influencent le plus les résultats, la profondeur restant à 3 malgré le changement des deux derniers paramètres pour les meilleurs TCC. On peut supposer que peu d’attributs suffisent à trouver la meilleure prédiction, le reste étant du bruit, il serait utile des les négliger dans nos modèles.


3.3.3 Perceptron Multi Couches

Paramètres modifiés: 
    1. Fonction d’activation (activation) : Identité, Tangente hyperbolique (tanh), Logistique, Unité de rectification linéaire (relu)
    2. Alpha : 0.1, 1, 10
    3. couches cachées et nombre de neurones y figurant (hidden_layer_sizes) : (5, 50), (5, 100), (10, 50), (10,100)
    4. Taux d’apprentissage : constant, invscaling, adaptif
    5.  solveur : lbfgs, sgd, adam



Datas : wisconsin.data



Pour le PMC nous obtenons les meilleurs résultats avec un alpha de 1 ou 10 et la fonction d’activation tangente hyperbolique le plus souvent et lbfgs comme solveur.
Dans notre cas, nombre de de couches cachées et le nombres de neurones y figurant influencent peu les résultats.
Je suppose que le alpha de 0.1 systématiquement plus mauvais dans notre cas est causé par la tendance à rester dans des minimas locaux moins bons lors de la descente de gradient.

Le TCC minimum obtenu est de 0.439 ce qui est mauvais. Nous pouvons déduire que la configuration des paramètres est d’autant plus important pour ce classificateur. Cependant, à part les trois paramètres cités avant, il est difficile de déterminer quelle est l’influence des autres paramètres car c’est sans doute leur combinaison qui explique cette variation de résultats.

Variation que nous pouvons observer si nous classons les modèles par ordre de TCC décroissant.
 

Nous pouvons remarquer un saut de 0,9 à 0,65 environ de TCC, en étudiant les données, la combinaison des autres paramètres avec la fonction logistique donne systématiquement des résultats moins bon, ainsi qu’une combinaison entre certains paramètres et le solveur invscaling.

Datas : dow_jownes_index.data


Il intéressant de noter la similarité des scores et des meilleurs paramètres malgré deux ensembles de données très différents.





On observe le même phénomène si l’on classe les modèles par TCC.





4. Conclusion :

	Scikitlearn fournit beaucoup d’outils pour améliorer ses modèles de classification ou de régression. Cependant il n’est pas toujours évident de comprendre l’influence des paramètres utilisés, une idée serait d’utiliser une régression sur ces données (donc du machine learning sur du machine learning) afin de déterminer les paramètres les plus influents en observant leurs poids. 
