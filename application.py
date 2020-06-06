from tkinter import *
from random import randint
import numpy as np
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import Pipeline
from pyearth import Earth
from PIL import Image, ImageTk
import matplotlib.gridspec as gridspec
import itertools
 
xlist = [[14,14]]
ylist = [0]
  
def app():
    # initialise a window.
    root = Tk()
    root.config(background='white')
    root.geometry("1000x700")
    
    lab = Label(root, text="Live Plotting", bg = 'white').pack()
    
    root.photo = ImageTk.PhotoImage(Image.open('foo.png'))
    vlabel = Label(root,image=root.photo)
    vlabel.pack()

    labx = Label(root, text="X", bg = 'white').pack()
    contentx = StringVar()
    ex = Entry(root,textvariable=contentx)
    ex.pack()
    laby = Label(root, text="Y", bg = 'white').pack()
    contenty = StringVar()
    ey = Entry(root,textvariable=contenty)
    ey.pack()
    labc = Label(root, text="Class", bg = 'white').pack()
    contentc = StringVar()
    color = Entry(root,textvariable=contentc)
    color.pack()
    

    def plotter():
        global xlist,ylist
        plt.clf()
        X = contentx.get()
        Y = contenty.get()
        colo = contentc.get()
        xlist.append([float(X),float(Y)])
        ylist.append(int(colo))
        
        contentx.set("")
        contenty.set("")
        contentc.set("")
        
        npx_list = np.array(xlist)
        npy_list = np.array(ylist)

        earth_classifier = Pipeline([('earth', Earth()),
                             ('logistic', LogisticRegression())])
        earth_classifier.fit(npx_list,npy_list)
        
        plot_decision_regions(npx_list, npy_list, clf=earth_classifier, legend=2)
        plt.savefig('foo.png')

        root.photo_n = ImageTk.PhotoImage(Image.open('foo.png'))
        vlabel.configure(image=root.photo_n)
        print("Image Updated")


        
    def gui_handler():
        plotter()

    def xorr_gui_handler():
        plt.clf()
        xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
                     np.linspace(-3, 3, 50))
        rng = np.random.RandomState(0)
        X = rng.randn(300, 2)
        y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0), 
                    dtype=int)
        earth_classifier = Pipeline([('earth', Earth(max_degree=3, penalty=1.5)),
                             ('logistic', LogisticRegression())])
        earth_classifier.fit(X,y)
        
        plot_decision_regions(X, y, clf=earth_classifier, legend=2)
        plt.savefig('foo.png')

        root.photo_n = ImageTk.PhotoImage(Image.open('foo.png'))
        vlabel.configure(image=root.photo_n)
        print("Image Updated")

    def cirr_gui_handler():
        plt.clf()
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
        earth_classifier = Pipeline([('earth', Earth(max_degree=3, penalty=1.5)),
                             ('logistic', LogisticRegression())])
        earth_classifier.fit(X,y)
        
        plot_decision_regions(X, y, clf=earth_classifier, legend=2)
        plt.savefig('foo.png')

        root.photo_n = ImageTk.PhotoImage(Image.open('foo.png'))
        vlabel.configure(image=root.photo_n)
        print("Image Updated")

    def moon_gui_handler():
        plt.clf()
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=100, random_state=123)
        earth_classifier = Pipeline([('earth', Earth(max_degree=3, penalty=1.5)),
                             ('logistic', LogisticRegression())])
        earth_classifier.fit(X,y)
        
        plot_decision_regions(X, y, clf=earth_classifier, legend=2)
        plt.savefig('foo.png')

        root.photo_n = ImageTk.PhotoImage(Image.open('foo.png'))
        vlabel.configure(image=root.photo_n)
        print("Image Updated")

    b = Button(root, text="Add", command=gui_handler, bg="red", fg="white")
    b.pack()
    
    xorr = Button(root, text="Xor dataset", command=xorr_gui_handler, bg="red", fg="white")
    xorr.pack()

    circle = Button(root, text="Circles dataset", command=cirr_gui_handler, bg="red", fg="white")
    circle.pack()

    moon = Button(root, text="Moon dataset", command=moon_gui_handler, bg="red", fg="white")
    moon.pack()
    
    root.mainloop()
 
if __name__ == '__main__':
    app()




