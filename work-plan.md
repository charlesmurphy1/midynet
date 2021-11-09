# Plan de travail du Zoo: 2021/11/14 - 2021/11/20

Ce plan est sujet à changer, soit au courant du Zoo ou avant. On énonce les objectifs du projet MIDyNet Phase II, et plus particulièrement ceux du zoo.

## Objectifs de MIDyNet II
Le but du projet est d'étudier les reconstructibilité des graphes à partir de séries temporelles du point de vue théorie. La question "Quand est-ce qu'un graphe est reconstructible?" est au coeur du projet et notre réponse fait intervenir le cadre théorique développer lors de MIDyNet I. Plus particulièrement, on voudra développer différent points présentés plus bas.

### Objectifs principaux
1. Présenter notre mesure de la reconstructibilité comme le coefficient d'incertitude S(G|X) = I(G;X) / H(G).
2. Présenter et utiliser les familles de graphes aléatoires \mathcal{G} = (G,\Theta) afin de généralisation de S(G|X) -> S(G,\Theta|X). Comme nous nous situerons dans le cadre de la reconstruction de réseaux, nous nous intéresserons plus particulière aux familles basées sur des hypothèses _non-informatives_.
3. Comparer la performance d'algorithmes de reconstruction de réseaux.
    a. Classifier les algorithmes de reconstruction en fonction des hypothèses faites.
    b. Déterminer une bonne méthode de comparaison des performance entre les algorithmes.
    c. Commenter la relation entre la performance de ces algorithmes et avec celle prédite par S(G,\Theta|X).
4. Étudier le comportement de la reconstructibilité S(G,\Theta|X):
    a. Dans les limites T -> inf et N -> inf.
    b. Pour différentes dynamiques X et familles (G,\Theta).
    c. Identifier les hypothèses structurelles à priori qui permettent les meilleures reconstructions dans ces différentes scénarios.

### Autres objectifs
Ces objectis supplémentaires pourront (si le temps le permet ou si on en resent le besoin) être exploré.
1. Étudier le comportement de I(\Theta;X) ainsi que de S(\Theta|X) = I(\Theta;X) / H(\Theta) et F(X|\Theta) = I(\Theta;X) / H(X).
2. Définir le rang de (G, \Theta) et quatifier son effet ainsi que celui des propiétés spectrales de (G, \Theta) sur la reconstructibilité.

## Plan de travail
Le Zoo se déroulera du dimanche 14 novembre au samedi 20 novembre. L'objectif spécifique de cette activité sera d'implémenter le code en C++ dont l'organigramme est disponible sur le repo [`code-organization`](./code-organization.pdf). Il existe une version Python de ce code qui pourra nous guider dans l'élaboration des abstractions, qu'il faudra néanmoins modifier 1) parce que l'architecture n'est pas parfaite et 2) elle sera difficile à traduire complètement en C++. Il y aura également d'autres tâches à faire / explorer durant la semaine. Voici donc un bref énoncé des tâches:
1. Les classes de base sont `RandomVariable`, `GraphPrior` et `Proposer`. Ces classes seront implémentées possiblement avant le commencement du Zoo.
2. Implémenter quelques `RandomVariable` élémentaires: `UniformVariable`, `PoissonVariable` et `ConstantVariable`. Ces classes seront aussi implémentées à l'avance et vont servir d'exemple sur comment implémenter des sous-classes de `RandomVariable`.
3. Implémenter les `GraphPrior` pour les familles de graphes élémentaires
    a. `BlockCountPrior`: classe virtuelle qui représente le nombre de blocs _B_ dans le SBM (et variants). On implémentera également des variantes Poisson(\bar{B}), Uniforme(1, N), etc.
    b. `NodeCountPrior`: classe virtuelle qui représente le vecteur _(N_r)_, où _N_r_ est le nombre de noeuds dans le bloc _r_ (hyperprior de `BlockPrior`). Comme on souhaite que tout _N_r > 0_, et que _\sum_r N_r = N_, on sait que l'ensemble _(N_r)_ est une composition de l'entier _N_ en _B_ parties. Donc, la version uniforme de _(N_r)_ nous demandera d'échantillonner uniformément les partitions de l'entier _N_ en _B_ parties.
    c. `BlockPrior`: classe qui représente le vecteur d'assignation des blocs _**b**_. Il y aura la variante de base où un objet de type `NodeCountPrior` devra être fournie et une variante uniforme où ce n'est pas le cas.
    d. `EdgeCountPrior`: classe virtuelle qui représente le nombre de liens total _E_. Comme pour `BlockCountPrior`, on implémentera des versions poisson et uniform.
    e. `EdgeMatrixPrior`: classe virtuelle qui représente la matrice symétrique _**e**_, où _e<sub>rs</sub>_ est le nombre de liens qui connecte les blocs _r_ et _s_. Pour le moment, il n'y aura qu'une version uniforme de cette classe. Notons que _sum<sub>r, s</sub> e<sub>rs</sub> = 2E_, donc les _{e<sub>rs</sub>|r < = s}_ sont en réalité des compositions faibles de l'entier _E_ en _B(B-1)/2_ parties. Ce sera à garder en tête.
    f. `DegreeCountPrior`: classe virtuelle qui représente le vecteur _(N_k)_, où _N_k_ est le nombre de noeuds ayant un degré _k_ (hyperprior de `DegreePrior`). De la même manière que pour _N_r_, on sait _\sum_k N_k = N_ mais ici il est possible que _N_k=0_. De plus, on a la condition additionnelle que _2E = \sum_k k N_k_. Ce faisant, _(N_k)_ est une partition de l'entier _2E_ en _N_ parties. On implémentera un version uniforme de cette variable aléatoire.
    g. `DegreePrior`: classe qui représente la séquence des degrés _**k** = (k_i)_. Comme pour `BlockPrior`, la variante de base sera définie par un objet `DegreeCountPrior` et on définiera une autre variante uniforme qui en héritera.
    h. Possiblement `LayerCountPrior`, `HierarchicalBlockPrior` et `HierarchicalEdgeMatrixPrior`: des versions hiérarchiques de _**b**_ -> _**b**<sup>(l)</sup>_ et _**e**_ -> _**e**<sup>(l)</sup>_ où _l\in[1, L]_ et _L_ est le nombre de couches. À voir si on a le temps et si ça semble utile (voir [[T. Peixoto, Phys. Rev. E **95**, 012317 (2017)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.012317)]).
4. Implémenter les différentes sous-classes de `Proposer`: `EdgeMoveProposer`, `BlockMoveProposer`, (il y en aura certainement d'autres, voir le code Python).
5. À partir de ces `GraphPrior`, implémenter `RandomGraph` et ses sous-classes, i.e. les différentes familles de graphes aléatoires, que nous utiliserons dans le projet:
    a. `DCSBMFamily`: classe qui représente la famille des _degree-corrected stochastic block models_, où _\Theta = {**b**, **e**, **k**}_. Cette classe possède les priors qui héritent de `BlockPrior`, de `EdgeMatrixPrior` et de `DegreePrior`.
    b. `SBMFamily`: classe quie représente la famille des _stochastic block models_, où _\Theta = {**b**, **e**}_. Essentiellement, cette classe ressemble beaucoup à `DCSBMFamily`, mais les méthodes `loglikelihood` et les priors seront différents. Notamment, cette classe ne possède pas de de membre `DegreePrior`.
    c. `ConfigurationFamily`: classe qui représente la famille des _configuration models_, où _\Theta = {**k**}_. Cette classe est identique à `DCSBMFamily`, à l'exception que _**b**_ est fixé de sorte que _b_i = 0_ pour tout _i_.
    d. `PlantedPartitionFamily`: à réfléchir.
    e. `ErdosRenyiFamily`: classe qui représente la famille des modèles de Erdos-Rényi, où _\Theta = {E}_. Cette classe est identique à `SBMFamily`, à l'exception que _**b**_ est fixé de sorte que _b_i = 0_ pour tout _i_.
    f. `UniformFamily`: classe qui représente la famille des graphes aléatoires de _N_ noeuds uniformément distribués, où _\Theta = {}_. Il existe une correspondence entre `UniformFamily` et `ErdosRenyiFamily`, où la variable _E_ doit être adéquatement choisi.
    Il y aura aussi, possiblement, d'autres familles de graphes aléatoires qui pourrait être utiles d'implémenter
    g. `LayeredConfigurationFamily`: classe qui représente la famille des _layered configuration model_, où _\Theta = {**l**, **k**}_ et _**l**_ est le vecteur des "couches". Ici, il est possible qu'on ait à jouer avec la notion de ce qu'est une couche (possiblement différente de celle qu'on définit à partir de la décomposition en oignon), puisqu'autrement je crois qu'on aura de la difficulté à écrire le log-likelihood et à définir le prior _**l**_.
    h. `CorrelatedConfigurationFamily`: classe qui représente la famille des _correlated configuration model_, où _\Theta = {**e**, **k**}_. Ce sera sans doute à rediscuter, mais je crois que la classe `CorrelatedConfigurationFamily` sera très similaires à la classe `DCSBMFamily`, mais où on aura fixé l'assignation des blocs au degrés des noeuds, i.e. _b_i = k_i_. Dans ce cas, _e_kl_ est le nombre de liens partagés entre des neuds de degrés k et l.
6. Implémenter `Dynamics` et ses sous-classes:
    a. `QNaryDynamics`: classe virtuelle qui représente une dynamique à _q_ états.
    b. `BinaryDynamics`: classe virtuelle qui représente une dynamique à 2 états, qui hérite de `QNaryDynamics` (possiblement pas nécessaire).
    c. Autres dynamiques.
7. Implémenter les générateurs de variables aléatoires nécessaires (voir l'[organigramme](./code-organization.svg)).
7. Écrire les _tests unitaires_ pour chaque classe avec `GTEST`.
8. Faire le _binding_ entre le code C++ et l'interface en Python.
9. Utiliser l'interface de `netrd` dans le module Python pour nous permettre d'utiliser les différentes mesures de reconstruction.
10. Développer la preuve pour la conjecture où, dans la limite quand T -> inf et N != inf, S(G|X) -> 1. Trouver un cas où ceci n'est pas vrai (vraisemblance non-unimodale?).
11. Explorer les limites T -> inf et N -> inf.
12. Explorer les régimes critiques (autour des transitions de phase).


## Autres idées
1. Proposer une méthode numérique pour calculer la reconstructibilité basée sur des données (G, X).
2. Proposer des approximations pour calculer analytiquement I(X;G), ce qui nous permettra potentiellement d'étudier le comportement asymptotique de I(G;X) dans la limite T->inf et/ou N->inf.
3. Proposer une formulation des mesures dans MIDyNet où les paramètres de la dynamique Phi sont également inconnus, possiblement basée sur I(X;Phi, Theta, G).
4. Proposer une formulation des mesures dans MIDyNet appliquée aux problèmes de plongement de graphes, basée sur I(X;G) où X=(x_i) sont les coordonées des noeuds dans un espace métrique.
5. [...]
