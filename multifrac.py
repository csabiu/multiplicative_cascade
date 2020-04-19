import numpy as np

def multifrac(pp, dim=2, size=2, levels=4, add_power=False): 
    S=size
    D=dim
    l=levels
    size=np.tile(S,D)
    add=0.0
    if (add_power):
        add=1.0

    pm=np.random.permutation(pp).reshape(size)
    for l in range(l):
        ng=S**(l+2)
        new=np.zeros(np.tile(ng,D))
    
        if (D==1):
            for i in range(0,ng,S):
                new[i:i+S]=np.random.permutation(pp).reshape(size)**((add*l+1)*1)
        elif (D==2):
            for i in range(0,ng,S):
                for j in range(0,ng,S):
                    new[i:i+S,j:j+S]=np.random.permutation(pp).reshape(size)**((add*l+1))        
        elif (D==3):
            for i in range(0,ng,S):
                for j in range(0,ng,S):
                    for k in range(0,ng,S):
                        new[i:i+S,j:j+S,k:k+S]=np.random.permutation(pp).reshape(size)**((add*l+1))
        else:
            print('dimension value not supported D=1,2,3')
            return 
            
        for dd in range(D):
            pm=pm.repeat(S, axis=dd)
        pm=pm*new
        
    fi=pp/np.sum(pp)
    if(add_power):
        Dth=np.log2(np.sum(fi**l))/(l+0.5) #this is a guess based on experiment 
    else:
        Dth=np.log2(np.sum(fi**2))
    print('power law slope (theory)=',Dth)

    return pm

def mock(density=[],boxsize=100,Npart=100):
    from scipy.stats import poisson
    dim=np.shape(np.shape(density))[0]
    Nmesh=np.shape(density)[0]
    ll=boxsize/Nmesh
    density = poisson.rvs(Npart*density/np.sum(density))
    
    i=0
    j=0
    k=0

    if (dim==3):
        xpoints=np.random.uniform(low=i*ll,high=(i+1)*ll,size=(density[i,j,k]))
        ypoints=np.random.uniform(low=j*ll,high=(j+1)*ll,size=(density[i,j,k]))
        zpoints=np.random.uniform(low=k*ll,high=(k+1)*ll,size=(density[i,j,k]))
        points=np.transpose(np.vstack((xpoints,ypoints,zpoints)))
    else:
        xpoints=np.random.uniform(low=i*ll,high=(i+1)*ll,size=(density[i,j]))
        ypoints=np.random.uniform(low=j*ll,high=(j+1)*ll,size=(density[i,j]))
        points=np.transpose(np.vstack((xpoints,ypoints)))
    for i in range(1,Nmesh):
        for j in range(Nmesh):
            if (dim==3):
                for k in range(Nmesh):
                    xpoints=np.random.uniform(low=i*ll,high=(i+1)*ll,size=(density[i,j,k]))
                    ypoints=np.random.uniform(low=j*ll,high=(j+1)*ll,size=(density[i,j,k]))
                    zpoints=np.random.uniform(low=k*ll,high=(k+1)*ll,size=(density[i,j,k]))
                    points=np.vstack((points,np.transpose(np.vstack((xpoints,ypoints,zpoints)))))
            else:
                xpoints=np.random.uniform(low=i*ll,high=(i+1)*ll,size=(density[i,j]))
                ypoints=np.random.uniform(low=j*ll,high=(j+1)*ll,size=(density[i,j]))
                points=np.vstack((points,np.transpose(np.vstack((xpoints,ypoints)))))
    print('number point samples:',np.shape(points)[0])

    if (dim==3):
        rans=np.random.uniform(low=0.0,high=boxsize,size=(10*Npart,3))
    if (dim==2):
        rans=np.random.uniform(low=0.0,high=boxsize,size=(10*Npart,2))
    
    return points, rans




