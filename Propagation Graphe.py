import numpy as np
import networkx as nx


class Environment_G (object) :

    def __init__(self,n = 30,k = 12,p_proba = 0.5,p_crea = 1,training = False, graph_type = 'w_s'):

        if graph_type == 'b_a' :
            self.graph = nx.barabasi_albert_graph(n,k)
            self.graph_type = 'b_a'

        else : # watts_strogatz' graph as default
            self.graph = nx.connected_watts_strogatz_graph(n,k,p_crea)
            self.graph_type = 'w_s'

        for edge in self.graph.edges :
            # Rq : poids := distance entre 2 individus, plus elle est grande moins il a de chance de se faire infecter
            self.graph.edges[edge]['weight'] = np.random.random()

        for node in self.graph :
            self.graph.nodes[node]['type'] = "susceptible"

        self.infected = []
        self.n_infect = 0
        self.taille = n
        self.proba = p_proba
        if not training :
            self.color = ['#1CC5F5' for k in range(self.taille)]
        self.training = training
        self.p_crea = p_crea
        self.k = k



    def get_neighborhood(self, node) :
        return list(self.graph.neighbors(node))

    def get_immediate_weight(self, node1, node2) :
        try :
            return self.graph.edges[node1,node2]['weight']
        except :
            return np.inf


    def susceptible(self, node) :
        return self.graph.nodes[node]['type'] == "susceptible"


    def get_weight_mat(self) :
        M = np.array([[np.inf for _ in range(self.taille)] for _ in range(self.taille)])
        for i in range(self.taille) :
            M[i][i] = 0
        for u,v in self.graph.edges :
            M[u][v] = self.graph.edges[u,v]['weight']
            M[v][u] = M[u][v]
        return M

    def get_adj_mat(self) :
        M = np.zeros((self.taille,self.taille), dtype = np.int8)
        for u,v in self.graph.edges :
            M[u,v] = 1
            M[v,u] = 1
        return M

    def set_asgraph(self, G, without = None) :

        self.graph = nx.Graph()
        self.graph.add_nodes_from(G.graph)
        for u,v in G.graph.edges :
            if u != without and v != without :
                self.graph.add_edge(u,v)
                self.graph.edges[u,v]["weight"] = G.graph.edges[u,v]["weight"]

        return


    #Fait se propager le virus autour des infectés
    def spread(self) :
        new_infected = []
        for node in self.infected :
            for voisin in self.graph.neighbors(node):
                if self.susceptible(voisin) :
                    if np.random.random() < self.proba*(1-self.graph.edges[node,voisin]["weight"]) :
                        self.graph.nodes[voisin]['type'] = 'infectious'
                        if not self.training :
                            self.color[voisin] = '#E33A0A'
                        new_infected += [voisin]
                        self.n_infect+=1

        self.infected += new_infected

        return new_infected



    #Initialise les premiers virus & tours d'avance du virus
    def start(self,nb = 1,tour_avance = 2) :
        assert nb < self.taille
        self.n_infect = nb
        for _ in range(nb):
            node = np.random.randint(self.taille)
            while not self.susceptible(node) :
                node = np.random.randint(self.taille)
            self.graph.nodes[node]['type'] = 'infectious'
            if not self.training :
                self.color[node] = '#E33A0A'
            self.infected += [node]

        for _ in range(tour_avance):
            self.spread()


    def is_finished(self) :
        for node in self.infected :
            for neighbor in self.graph.neighbors(node) :
                if self.susceptible(neighbor) :
                    return False
        return True

    def results(self) :
        return self.n_infect


    def clear(self) :
        self.graph = nx.generators.random_graphs.connected_watts_strogatz_graph(self.taille, self.k, self.p_crea)

        for edge in self.graph.edges :
            self.graph.edges[edge]['weight'] = np.random.random()

        for node in self.graph :
            self.graph.nodes[node]['type'] = 'susceptible'

        self.infected = []
        self.n_infect = 0
        if not self.training :
            self.color = ['#1CC5F5' for k in range(self.taille)]





    #Permet au joueur de jouer à son tour
    def step(self, node) :
        try :
            if self.susceptible(node) :
                self.graph.nodes[node]['type'] = 'vaccinated'
            else :
                raise Exception
        except :
            raise Exception("A problem occured, please verify the node is neither infected nor vaccinated")




    #affiche le graphe et joue
    def jeu_gameplay(self, n_ini = 1, n_avance = 0, IA = None) :
        assert not self.training

        self.clear()
        self.start(n_ini,n_avance)
        is_ended = False


        fig, ax = plt.subplots()
        fig.set_size_inches((19,9))
        plt.subplots_adjust(bottom=0.2)
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_edges(self.graph, pos=pos, ax=ax)
        nx.draw_networkx_nodes(self.graph, pos=pos, node_color=self.color,  ax=ax)
        nx.draw_networkx_labels(self.graph, pos=pos, labels=dict(zip(list(range(len(self.graph))),list(range(len(self.graph))))), ax=ax)
        if IA is None :
            IA = lambda x : np.random.randint(self.taille)




        def finish() :
            ax.clear()
            nx.draw_networkx_edges(self.graph, pos=pos, ax=ax)
            nx.draw_networkx_nodes(self.graph, pos=pos, node_color=self.color,  ax=ax)
            nx.draw_networkx_labels(self.graph, pos=pos, labels=dict(zip(range(len(self.graph)),range(len(self.graph)))), ax=ax)


            plt.text(2.2,6.5, "Félicitation" if self.n_infect<=self.taille/2 else "Dommage", horizontalalignment = 'center', verticalalignment = 'center', color = 'yellow', fontsize = 60, alpha = 0.9, bbox=dict(facecolor='red', alpha=0.7,boxstyle = 'round4'))
            ax.set_title(f'{self.taille-self.n_infect} personnes sauvées')



        def help(x) :
            try :
                node = IA(self)
                self.step(node)
                self.color[node] = '#C5F51C'
                if self.is_finished() :
                    finish()

                else :
                    nx.draw_networkx_edges(self.graph, pos=pos, ax=ax)
                    nx.draw_networkx_nodes(self.graph, pos=pos, node_color=self.color, ax=ax)
                    nx.draw_networkx_labels(self.graph, pos=pos, labels=dict(zip(range(len(self.graph)),range(len(self.graph)))), ax=ax)

            except :
                pass #cela correspond aux entrées typique "" par défaut


        new_button3 = plt.axes([0.41, 0.05, 0.1, 0.075])
        button3 = Button(new_button3, 'Help', color='.90', hovercolor='1')
        cid1 = button3.on_clicked(help)



        def update(x):
            if not x.isnumeric() :
                pass

            try :
                if self.is_finished() :
                    finish()
                else :
                    ax.clear()
                    y = int(x)
                    if y>self.taille or y<0 or self.color[y] != '#1CC5F5' :
                        raise Exception('LengthError Taille incorrecte')
                    self.color[y] = '#C5F51C'
                    self.graph.nodes[y]['type'] = 'vaccinated'

                    if self.is_finished() :
                        finish()

                    else :
                        nx.draw_networkx_edges(self.graph, pos=pos, ax=ax)
                        nx.draw_networkx_nodes(self.graph, pos=pos, node_color=self.color, ax=ax)
                        nx.draw_networkx_labels(self.graph, pos=pos, labels=dict(zip(range(len(self.graph)),range(len(self.graph)))), ax=ax)

                        text.set_val("")

            except :
                ax.clear()
                nx.draw_networkx_edges(self.graph, pos=pos, ax=ax)
                nx.draw_networkx_nodes(self.graph, pos=pos, node_color=self.color,  ax=ax)
                nx.draw_networkx_labels(self.graph, pos=pos, labels=dict(zip(range(len(self.graph)),range(len(self.graph)))), ax=ax)
                text.set_val("")




        def infect(x) :
            if self.is_finished() :
                finish()

            else :
                ax.clear()
                new_infected = self.spread()
                for new_pos in new_infected :
                    self.color[new_pos] = '#E33A0A'

                if self.is_finished() :
                    finish()

                else :
                    nx.draw_networkx_edges(self.graph, pos=pos, ax=ax)
                    nx.draw_networkx_nodes(self.graph, pos=pos, node_color=self.color,  ax=ax)
                    nx.draw_networkx_labels(self.graph, pos=pos, labels=dict(zip(range(len(self.graph)),range(len(self.graph)))), ax=ax)



        posbutton = plt.axes([0.81, 0.05, 0.1, 0.075])
        button = Button(posbutton, 'Infecter', color='.90', hovercolor='1')
        cid_butt = button.on_clicked(infect)




        postxtbox = plt.axes([0.7, 0.05, 0.1, 0.075]) #position sur l'affichage
        
        # textbox placé en (postxtbox) sur l'écran, se nommant vacciner, sans valeur initiale, couleur initiale, couleur quand la souris touche, 
        # distance label et côté droit du textbox
        text = TextBox(postxtbox, 'vacciner' , initial='', color='.90', hovercolor='1', label_pad=0.01) 

        cid_text = text.on_submit(update) #change quand on clique sur entrée (sinon on_text_change()

        def restart(x) :
            plt.close(fig)
            self.jeu_gameplay(n_ini, n_avance, IA)
        def leave(x) :
            plt.close(fig)

        new_button1 = plt.axes([0.19, 0.05, 0.1, 0.075])
        button1 = Button(new_button1, 'Leave', color='.90', hovercolor='1')
        cid1 = button1.on_clicked(leave)

        new_button2 = plt.axes([0.3, 0.05, 0.1, 0.075])
        button2 = Button(new_button2, 'Restart', color='.90', hovercolor='1')
        cid2 = button2.on_clicked(restart)





        plt.show()






## Application : recherche de stratégie 






## Données
import matplotlib.pyplot as plt
import time

# Paramètres relatifs au jeu
n, k, p_proba, p_crea, graph_type = 30, 10, 0.3, 0.1, 'w_s'
nb_vaccin, nb_infect_ini, avance = 2, 1, 1

# Paramètres relatifs à la population
mu, n_gene, pop_size = 10, 6, 6

# Paramètres relatifs à l'amélioration de la population
n_eval,n_steps, eps, eps_decr, eps_min  = 100, 5, 0.8, 0.97, 0.3
N = (pop_size+1)//2


# Proporsion de la population sauvée pour considérer le résultat comme satisfaisant
condition = 0.7

# Faire la moyenne sur les n_average premiers individus
n_average = pop_size

# Création de la population aléatoire initiale
pop = np.random.randint(-mu,mu+1, size = (pop_size,n_gene))
# Garder en mémoire le meilleur
best_gene = (0, np.zeros(6,dtype = int))

# Mémoire
episode_count = 0 ; mean_vict = 0
history_reward = []


## Import/Save values

import sqlite3 # module

def import_val() :
    """ return the list of tuples : (step, best_val,
                                    best_gene, epsilon) """

    bd = sqlite3.connect(path)
    c = bd.cursor()
    L = c.execute("SELECT * FROM Valeurs;").fetchall()
    bd.close()
    return L




def save_best_val(step, best_val, best_gene, epsilon) :


    bd = sqlite3.connect(path)
    c = bd.cursor()
    c.execute(f"INSERT INTO Valeurs (step, best_val, best_gene, epsilon) VALUES {(step, best_val, str(best_gene), epsilon)};")
    bd.commit()
    bd.close()





## Algorithmes utiles


def floydwarsahll (G) :
    """ distance du plus court chemin """
    M = G.get_weight_mat()
    for k in range(G.taille) :
        for i in range(G.taille) :
            for j in range(G.taille) :
                M[i][j] = min(M[i][j], M[i][k]+M[k][j]+1)
    return M



def floydwarsahll_path(G) :
    """ Distance et chemin du plus court chemin """
    M = G.get_weight_mat()
    suivant = [[k for k in range(G.taille)] for _ in range(G.taille) ]
    for k in range(G.taille) :
        for i in range(G.taille) :
            for j in range(G.taille) :
                if M[i][k]+M[k][j] +1 < M[i][j] :
                    M[i][j] = M[i][k]+M[k][j]+1
                    suivant[i][j] = suivant[i][k]
    return M,suivant




def argmax(l) :
    """ renvoie l'argument de la plus grande valeure non nulle """
    temp = -1
    n = len(l)
    for k in range(n) :
        if l[k] != 0 :
            temp = k
            break
    if temp<0 :
        raise Exception('Zero or Empty List')
    for k in range(temp+1, n) :
        if l[k] != 0 and l[k]>l[temp] :
            temp = k
    return temp




def tri_insertion(l):
    """ On s'attend à ce qu'il n'y ait pas grand changements d'odre (les premiers le resteront en général)
    Cet algorithme a beau avoir une compléxité O(n^2) il est cependant linéaire pour les listes déjà triées :) """
    for i in range(1,len(l)):
        val = L[i]
        j = i-1
        while j>=0 and L[j] < val:
            L[j+1] = L[j] # decalage
            j = j-1
        L[j+1] = val





## Centralités à utiliser

def IA_Dense(G):
    """prend celui avec le plus de voisins (comptés
    avec proximité)"""
    possibilites = np.zeros(G.taille)
    for node in G.infected :
        for neighbor in G.get_neighborhood(node) :
            if G. susceptible(neighbor) :
                possibilites[neighbor] += G.get_immediate_weight(node,neighbor)
    return possibilites


def IA_Degre(G) :
    """prend le susceptible ayant le plus de voisins infectés"""
    possibilites = np.zeros(G.taille)
    for k in range(G.taille) :
        if G.susceptible(k) :
            temp = 0
            for neigh in G.get_neighborhood(k) :
                if G.is_infected(neigh) :
                    temp+=1
            possibilites[k] = temp
    return possibilites


def IA_Shortest_Path(G) :
    """prend le susceptible le plus proche d'un infecté"""
    W = floydwarsahll(G)
    possibilites = np.zeros(G.taille)
    for node in G.infected :
        for j in range(G.taille) :
            if G. susceptible(j) :
                possibilites[j] = max(possibilites[j], W[node][j])

    return possibilites



import copy
def IA_Vitality(G) :

    possibilites = np.zeros(G.taille,dtype = np.float16)

    G_ = copy.deepcopy(G)
    for node in G.graph :
        if G_.graph.nodes[node]["type"] == "vaccinated" :
            G_.graph.remove_node(node)

    M = floydwarsahll(G_)
    for i in range(G.taille):
        for j in range(G.taille) :
            if M[i,j] == np.inf :
                M[i,j] = G.taille
    m = np.sum(M)
    g_ = Environment_G(n = G_.taille, k = G_.k)
    for node in G.graph.nodes :
        if G.susceptible(node) :
            g_.set_asgraph(G_, without = node)
            M2 = floydwarsahll(g_)
            for i in range(G_.taille):
                for j in range(G_.taille) :
                    if M2[i,j] == np.inf :
                        M2[i,j] = G.taille
            possibilites[node] = np.abs(np.sum(M2)-m)
    return np.argmax(possibilites)




## IA Commune

# Pas optimisée : une meilleure forme serait de tout réunir autour d'une seule boucle for au lieu de faire les mêmes répétitions... 
# Cette version est cependant plus claire donc je la laisse

def IA_common(l,G) :

    dens = IA_Dense(G)
    deg = IA_Degre(G)
    sp = IA_Shortest_Path(G)
    vit = IA_Vitality(G)

    value = np.dot(l,[dens,deg,sp,prox,inter,pr,vit])
    return argmax(value)





## Evaluation de l'application G->IA_common(l,G) pour chaque l

def evaluate(l, n, k, p_proba, p_crea, graph_type, n_eval, nb_vaccin, nb_infect_ini, avance) :

    G = Environment_G(n = n, k = k, p_proba = p_proba, p_crea = p_crea, training = True, graph_type = graph_type)

    rez = (0,0)
    for i in range(n_eval) :
        G.clear()
        G.start(nb_infect_ini, avance)
        while not G.is_finished() :
            for _ in range(nb_vaccin) :
                try :
                    G.step(IA_common(l,G))
                except :
                    pass
            G.spread()

        x = 1 if G.n_infect<(G.taille/2) else 0
        rez = (rez[0]+x, rez[1] + G.taille-G.n_infect)


    return (rez[0] / n_eval, rez[1] / n_eval)


def jeu_IA(IA, G, nb_vaccin) :

    while not G.is_finished() :
        for _ in range(nb_vaccin) :
            try :
                G.step(IA(G))
            except :
                pass
        G.spread()
    return G.taille-G.n_infect



## Algorithme genetique

assert n_average <= pop_size

#initialisation des données


# Faire tourner tant que ce n'est pas résolu
for episode_count in range(1,n_steps+1) :

    # Evaluer la population
    vict = [(evaluate(pop[i], n, k, p_proba, p_crea, graph_type, n_eval, nb_vaccin, nb_infect_ini, avance),i) for i in range(pop_size)]

    # Classer la population
    tri_insertion(vict,0,pop_size-1)
    pop = [pop[i] for _,i in vict]


    # Sauvegarder un historique d'action
    mean_vict = np.mean(vict[:n_average], axis = 0)[0]
    history_reward.append([mean_vict,vict[0][0],eps])

    save_best_val(episode_count,vict[0][0],pop[0],eps)
    # Faire progresser les critères de sélection
    eps *= eps_decr
    eps = max(eps, eps_min)


    # Sauvegarder les meilleurs genes
    if vict[0][0]>best_gene[0] :
        best_gene = (vict[0][0], pop[vict[0][1]])


    print( f'Fin du tour {episode_count}/{n_steps} : \n{round(mean_vict,2)} personnes sauvées en moyenne \n{round(vict[0][0],2)} : meilleur score \n{len([vict2 for vict2 in vict if vict2[0]>=n/2])} individu(s) en a (ont) sauvé plus de la moitié \n' )


    if mean_vict > n*condition or episode_count == n_steps :
        break


    # Repeupler la population à partir de la meilleure moitiée
    for i in range(N) :
        v1, v2 = np.random.choice(range(N), size=2, replace = False)
        split = np.random.randint(n_gene)
        pop[-i-1] = pop[v1][:split] + pop[v2][split:]

    # Faire muter aléatoirement une partie de la population
    for i in range(pop_size) :
        if np.random.rand() < eps :
            pop[i][np.random.randint(n_gene)] = np.random.randint(-mu,mu+1)




print(pop[:n_average])

