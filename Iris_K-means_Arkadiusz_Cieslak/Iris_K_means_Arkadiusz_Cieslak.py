import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from array import array
from sklearn.cluster import KMeans
import pandas as pd



def RozdzielAtrybuty(sepalL,sepalW,petalL,petalW):
    for i in range(150):
        sepalL.append(float(iris[i][0]))
        sepalW.append(float(iris[i][1]))
        petalL.append(float(iris[i][2]))
        petalW.append(float(iris[i][3]))
def Histogram3Atrybuty(x,y,z):
    formatting = dict(histtype="barstacked", alpha=0.5,density=True,stacked=True)
    plt.grid(True)
    plt.hist(x,**formatting,label="Sepal Length")
    plt.hist(y,**formatting,label="Sepal Width")
    plt.hist(z,**formatting,label="Petal Width")
    plt.xlabel("Wartość atrybutu")
    plt.ylabel("Ilość wystąpień atrybutu")
    plt.legend(loc="upper right")
    plt.show()
def Scatter3Atrybuty():
    ax=plt.axes(projection="3d")
    ax.scatter(sepalL[0:49],sepalW[0:49],petalL[0:49],color="red",label="Iris Setosa")
    ax.scatter(sepalL[49:99],sepalW[49:99],petalL[49:99],color="green",label="Iris Versicolor")
    ax.scatter(sepalL[99:149],sepalW[99:149],petalL[99:149],color="black",label="Iris Virginica")
    ax.legend(loc="upper right")
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Petal Length')
    ax.set_zlabel('Sepal Width')
    plt.show()
def Kmeans2Atrybuty():
    
    inp=""
    inp=input("Wybierz dla jakich atrybutów chcesz uruchomić algorytm:\n"
          "1.Sepal Length & Sepal Width\n"
          "2.Sepal Length & Petal Length\n"
          "3.Sepal Length & Petal Width\n"
          "4.Sepal Width & Petal Length\n"
          "5.Sepal Width & Petal Width\n"
          "6.Petal Length & Petal Width\n"
        )
    if inp =="1":
        km=KMeans(n_clusters=3)
        y=km.fit_predict(df[["SepalL","SepalW"]])
        df["cluster"]=y
        df.head()
        km.cluster_centers_
        df1 = df[df.cluster==0]
        df2 = df[df.cluster==1]
        df3 = df[df.cluster==2]
        plt.scatter(df1.SepalL,df1.SepalW,color="blue",label="Iris Setosa")
        plt.scatter(df2.SepalL,df2.SepalW,color="red",label="Iris Versicolor")
        plt.scatter(df3.SepalL,df3.SepalW,color="yellow",label="Iris Virginica")
        plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="Black",marker='*',label='centroid',s=300)
        plt.legend(loc="upper right")
        plt.xlabel('Sepal length')
        plt.ylabel('Petal Length')
        plt.show()
    elif inp=="2":
        km=KMeans(n_clusters=3)
        y=km.fit_predict(df[["SepalL","PetalL"]])
        df["cluster"]=y
        df.head()
        km.cluster_centers_
        df1 = df[df.cluster==0]
        df2 = df[df.cluster==1]
        df3 = df[df.cluster==2]
        plt.scatter(df1.SepalL,df1.PetalL,color="blue",label="Iris Setosa")
        plt.scatter(df2.SepalL,df2.PetalL,color="red",label="Iris Versicolor")
        plt.scatter(df3.SepalL,df3.PetalL,color="yellow",label="Iris Virginica")
        plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="Black",marker='*',label='centroid',s=300)
        plt.legend(loc="upper right")
        plt.xlabel('Sepal length')
        plt.ylabel('Petal Length')
        plt.show()
    elif inp=="3":
        km=KMeans(n_clusters=3)
        y=km.fit_predict(df[["SepalL","PetalW"]])
        df["cluster"]=y
        df.head()
        km.cluster_centers_
        df1 = df[df.cluster==0]
        df2 = df[df.cluster==1]
        df3 = df[df.cluster==2]
        plt.scatter(df1.SepalL,df1.PetalW,color="blue",label="Iris Setosa")
        plt.scatter(df2.SepalL,df2.PetalW,color="red",label="Iris Versicolor")
        plt.scatter(df3.SepalL,df3.PetalW,color="yellow",label="Iris Virginica")
        plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="Black",marker='*',label='centroid',s=300)
        plt.legend(loc="upper right")
        plt.xlabel('Sepal length')
        plt.ylabel('Petal Width')
        plt.show()
    elif inp=="4":
        km=KMeans(n_clusters=3)
        y=km.fit_predict(df[["SepalW","PetalL"]])
        df["cluster"]=y
        df.head()
        km.cluster_centers_
        df1 = df[df.cluster==0]
        df2 = df[df.cluster==1]
        df3 = df[df.cluster==2]
        plt.scatter(df1.SepalW,df1.PetalL,color="blue",label="Iris Setosa")
        plt.scatter(df2.SepalW,df2.PetalL,color="red",label="Iris Versicolor")
        plt.scatter(df3.SepalW,df3.PetalL,color="yellow",label="Iris Virginica")
        plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="Black",marker='*',label='centroid',s=300)
        plt.legend(loc="upper right")
        plt.xlabel('Sepal Width')
        plt.ylabel('Petal Length')
        plt.show()
    elif inp=="5":
        km=KMeans(n_clusters=3)
        y=km.fit_predict(df[["SepalW","PetalW"]])
        df["cluster"]=y
        df.head()
        km.cluster_centers_
        df1 = df[df.cluster==0]
        df2 = df[df.cluster==1]
        df3 = df[df.cluster==2]
        plt.scatter(df1.SepalW,df1.PetalW,color="blue",label="Iris Setosa")
        plt.scatter(df2.SepalW,df2.PetalW,color="red",label="Iris Versicolor")
        plt.scatter(df3.SepalW,df3.PetalW,color="yellow",label="Iris Virginica")
        plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="Black",marker='*',label='centroid',s=300)
        plt.legend(loc="upper right")
        plt.xlabel('Sepal Width')
        plt.ylabel('Petal Width')
        plt.show()
    elif inp=="6":
        km=KMeans(n_clusters=3)
        y=km.fit_predict(df[["PetalL","PetalW"]])
        df["cluster"]=y
        df.head()
        km.cluster_centers_
        df1 = df[df.cluster==0]
        df2 = df[df.cluster==1]
        df3 = df[df.cluster==2]
        plt.scatter(df1.PetalL,df1.PetalW,color="blue",label="Iris Setosa")
        plt.scatter(df2.PetalL,df2.PetalW,color="red",label="Iris Versicolor")
        plt.scatter(df3.PetalL,df3.PetalW,color="yellow",label="Iris Virginica")
        plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="Black",marker='*',label='centroid',s=300)
        plt.legend(loc="upper right")
        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')
        plt.show()
    else:
        print("Wybrałeś złą opcję, program kończy działanie.\n")
        quit()
    blad=0
    float(blad)
    for i in range(0,50):
        if (df.cluster[0]!=df.cluster[i]):
            blad=blad+1
    for i in range(50,100):
        if (df.cluster[50]!=df.cluster[i]):
            blad=blad+1
    for i in range(100,150):
        if (df.cluster[100]!=df.cluster[i]):
            blad=blad+1
    a=100-((100*blad)/150)
    print(f"Algorytm sprawdził się w:{a:.2f}%")

def Kmeans3Atrybuty():
    inp=""
    inp=input("Wybierz dla jakich atrybutów chcesz uruchomić algorytm:\n"
          "1.Sepal Length, Sepal Width, Petal Length\n"
          "2.Sepal Width, Petal Length, Petal Width\n"
          "3.Petal Length, Petal Width, Sepal Length\n"
          "4.Sepal Length, Sepal Width, Petal Width\n"
        )
    if inp=="1":
        km=KMeans(n_clusters=3)
        y=km.fit_predict(df[["SepalL","SepalW","PetalL"]])
        df["cluster"]=y
        df.head()
        km.cluster_centers_
        df1 = df[df.cluster==0]
        df2 = df[df.cluster==1]
        df3 = df[df.cluster==2]
        ax=plt.axes(projection="3d")
        ax.scatter(df1.SepalL,df1.SepalW,df1.PetalL,color="blue",label="Iris Setosa")
        ax.scatter(df2.SepalL,df2.SepalW,df2.PetalL,color="red",label="Iris Versicolor")
        ax.scatter(df3.SepalL,df3.SepalW,df3.PetalL,color="yellow",label="Iris Virginica")
        ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],km.cluster_centers_[:,2],color="Black",marker='*',label='centroid',s=300)
        ax.legend(loc="upper right")
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Petal Length')
        ax.set_zlabel('Sepal Width')
        plt.show()
    elif inp=="2":
        km=KMeans(n_clusters=3)
        y=km.fit_predict(df[["SepalW","PetalL","PetalW"]])
        df["cluster"]=y
        df.head()
        km.cluster_centers_
        df1 = df[df.cluster==0]
        df2 = df[df.cluster==1]
        df3 = df[df.cluster==2]
        ax=plt.axes(projection="3d")
        ax.scatter(df1.SepalW,df1.PetalL,df1.PetalW,color="blue",label="Iris Setosa")
        ax.scatter(df2.SepalW,df2.PetalL,df2.PetalW,color="red",label="Iris Versicolor")
        ax.scatter(df3.SepalW,df3.PetalL,df3.PetalW,color="yellow",label="Iris Virginica")
        ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],km.cluster_centers_[:,2],color="Black",marker='*',label='centroid',s=300)
        ax.legend(loc="upper right")
        ax.set_xlabel('Sepal Width')
        ax.set_ylabel('Petal Length')
        ax.set_zlabel('Petal Width')
    elif inp=="3":
        km=KMeans(n_clusters=3)
        y=km.fit_predict(df[["PetalL","PetalW","SepalL"]])
        df["cluster"]=y
        df.head()
        km.cluster_centers_
        df1 = df[df.cluster==0]
        df2 = df[df.cluster==1]
        df3 = df[df.cluster==2]
        ax=plt.axes(projection="3d")
        ax.scatter(df1.PetalL,df1.PetalW,df1.SepalL,color="blue",label="Iris Setosa")
        ax.scatter(df2.PetalL,df2.PetalW,df2.SepalL,color="red",label="Iris Versicolor")
        ax.scatter(df3.PetalL,df3.PetalW,df3.SepalL,color="yellow",label="Iris Virginica")
        ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],km.cluster_centers_[:,2],color="Black",marker='*',label='centroid',s=300)
        ax.legend(loc="upper right")
        ax.set_xlabel('Petal Length')
        ax.set_ylabel('Petal Width')
        ax.set_zlabel('Sepal Length')
    elif inp=="4":
        km=KMeans(n_clusters=3)
        y=km.fit_predict(df[["SepalL","SepalW","PetalW"]])
        df["cluster"]=y
        df.head()
        km.cluster_centers_
        df1 = df[df.cluster==0]
        df2 = df[df.cluster==1]
        df3 = df[df.cluster==2]
        ax=plt.axes(projection="3d")
        ax.scatter(df1.SepalL,df1.SepalW,df1.PetalW,color="blue",label="Iris Setosa")
        ax.scatter(df2.SepalL,df2.SepalW,df2.PetalW,color="red",label="Iris Versicolor")
        ax.scatter(df3.SepalL,df3.SepalW,df3.PetalW,color="yellow",label="Iris Virginica")
        ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],km.cluster_centers_[:,2],color="Black",marker='*',label='centroid',s=300)
        ax.legend(loc="upper right")
        ax.set_xlabel('Sepal Length')
        ax.set_ylabel('Sepal Width')
        ax.set_zlabel('Petal Width')
    else:
        print("Wybrałeś złą opcję, program kończy działanie.\n")
        quit()
    blad=0
    float(blad)
    for i in range(0,50):
        if (df.cluster[0]!=df.cluster[i]):
            blad=blad+1
    for i in range(50,100):
        if (df.cluster[50]!=df.cluster[i]):
            blad=blad+1
    for i in range(100,150):
        if (df.cluster[100]!=df.cluster[i]):
            blad=blad+1
    a=100-((100*blad)/150)
    print(f"Algorytm sprawdził się w:{a:.2f}%")
def Kmeans4Atrybuty():
    km=KMeans(n_clusters=3)
    y=km.fit_predict(df[["SepalL","SepalW","PetalL","PetalW"]])
    df["cluster"]=y
    df.head()
    blad=0
    float(blad)
    for i in range(0,50):
        if (df.cluster[0]!=df.cluster[i]):
            blad=blad+1
    for i in range(50,100):
        if (df.cluster[50]!=df.cluster[i]):
            blad=blad+1
    for i in range(100,150):
        if (df.cluster[100]!=df.cluster[i]):
            blad=blad+1
    a=100-((100*blad)/150)
    print(f"Algorytm sprawdził się w:{a:.2f}%")
def Menu():
    n=""    
    print(
        "Grupowanie danych Iris metodą k-średnich\n"
        "Menu:\n"
        "1.Wygeneruj 3 atrybutowy histogram danych\n"
        "2.Wygeneruj 3 atrybutowy scatter danych\n"
        "3.Zastosuj algorytm K-means dla 2 atrybutów, wygeneruj odpowiadający mu wykres i pokaż procentowe porównanie wygenerowanych danych do oryginalnych\n"
        "4.Zastosuj algorytm K-means dla 3 atrybutów, wygeneruj odpowiadający mu wykres i pokaż procentowe porównanie wygenerowanych danych do oryginalnych\n"
        "5.Zastosuj algorytm K-means dla 4 atrybutów i pokaż procentowe porównanie wygenerowanych danych do oryginalnych\n"
        "6.Zakończ\n"
        )
    while n!=6:
    
        n=input("Wybierz opcję\n")
        if n=="1":
            Histogram3Atrybuty(sepalL,sepalW,petalL)
        elif n=="2":
            Scatter3Atrybuty()
        elif n=="3":
            Kmeans2Atrybuty()
        elif n=="4":
            Kmeans3Atrybuty()
        elif n=="5":
            Kmeans4Atrybuty()
        elif n=="6":
            print("Program kończy działanie")
            quit()
        else:
            print("Proszę podać odpowiednią opcję")



#Otworzenie pliku i zczytanie calosci jako objekt
with open("iris.data", 'r') as f:
    iris = np.loadtxt("iris.data", dtype="object",delimiter=',')
#utworzenie data framów do K-means
df = pd.read_csv("iris2.data")
df.head()
#stworzenie tablic dla danych typu float
sepalL=array("f")
sepalW=array("f")
petalL=array("f")
petalW=array("f")
#przypisanie do stworzonych wczesniej tablic odpowiednich wartosci
RozdzielAtrybuty(sepalL,sepalW,petalL,petalW)
#Menu
Menu()
df.close
