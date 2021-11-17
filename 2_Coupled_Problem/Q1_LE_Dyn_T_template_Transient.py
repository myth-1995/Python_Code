import numpy as np

# This is an element template for PyFEMP

def Elmt_Init():
    '''
    The Elmt_Init() function is used to introduce the element to PyFEMP.
    '''
    NoElementDim         = 2                   # number of dimensions
    NoElementNodes       = 4                   # number of nodes of this element
    NoElementHistory     = 4*3*3               # number of scalar history parameters
    ElementDofNames      = ["UX", "UY", "DT"]  # list with string names for each dof
    # list with string name for material paramter
    ElementMaterialNames = ["E", "nu", "rho", "eta", "bx", "by", "a_q", "c", "r", "T0", "a_T"]  
    # list with string name for postprocessing
    ElementPostNames     = ["UX", "UY", "T", "SigMises"]   
    return NoElementDim, NoElementNodes, ElementDofNames, NoElementHistory, ElementMaterialNames, ElementPostNames


def Elmt_KS(XL, UL, Hn, Ht, Mat, dt):
    '''
    The Elmt_KS(XL, UL, Hn, Ht, Mat, dt) function returns the element vector and matrix.
    Input: 
            XL  = [x11, x12, x21, x22, ..., xn2]    -> np.vector of nodal coordinates for n-Nodes in 2d  
            UL  = [u11, u12, u21, u22, ..., un2]    -> np.vector of current nodal dofs for n-Nodes and 2 dofs at each
            Hn  = [h1, h2,... hm]                   -> np.vector for the previous history of length m (! do NOT write in here)
            Ht  = [h1, h2,... hm]                   -> np.vector for the updated history of length m (!write in here)
            Mat = [M1, M2, ... Mo]                  -> list of material parameter in the order defined in Elmt_Init()
            dt  = dt                                -> time step size
    '''
    # initialize element vector /matrix
    r_e = np.zeros(4*3)
    k_e = np.zeros((4*3, 4*3))

    # read in material parameter
    Emod, nu, rho, mu_eta, bx, by, a_q, c, r, T0, a_T = Mat
    b = np.array([bx, by])

    # restructure input to fit our notation
    xI = np.array([[XL[0], XL[1]], [XL[2], XL[3]], [XL[4], XL[5]], [XL[6], XL[7]]])
    uI = np.array([[UL[0], UL[1]], [UL[3], UL[4]], [UL[6], UL[7]], [UL[9], UL[10]]])
    DTI = np.array([UL[2], UL[5], UL[8], UL[11]])

    # read histroy - Newmark time integration
    aIn    = np.array([[Hn[0],  Hn[1] ], [Hn[2] , Hn[3] ], [Hn[4] , Hn[5] ], [Hn[6] , Hn[7]] ])
    vIn    = np.array([[Hn[8],  Hn[9] ], [Hn[10], Hn[11]], [Hn[12], Hn[13]], [Hn[14], Hn[15]]])
    uIn    = np.array([[Hn[16], Hn[17]], [Hn[18], Hn[19]], [Hn[20], Hn[21]], [Hn[22], Hn[23]]])
    ddDTIn = np.array([Hn[24+0], Hn[24+1], Hn[24+2 ], Hn[24+3 ]])
    dDTIn  = np.array([Hn[24+4], Hn[24+5], Hn[24+6 ], Hn[24+7 ]])
    DTIn   = np.array([Hn[24+8], Hn[24+9], Hn[24+10], Hn[24+11]])

    #computing bulk modulus

    kappa = 0.0

    kappa = Emod / (3*(1-(2*nu)))




    # compute current acceleration and velocity - Newmark time integration

    # Compute acceleration using Newmark's Method
    aI = np.zeros((4, 2))
    for I in range(4):
        for i in range(2):
            aI[I, i] = ((4 / dt ** 2) * (uI[I, i] - uIn[I, i] - (vIn[I, i] * dt))) - aIn[I, i]

    # Computing the Velocity Using Newmark's Method
    vI     = np.zeros((4,2))
    for I in range(4):
        for i in range(2):
            vI[I, i] = ((2/dt)*(uI[I, i] - uIn[I, i])) - vIn[I, i]


    # Coding second derivative of Temperature
    ddDTI = np.zeros(4)

    for I in range(4):
        ddDTI[I] = ((4.0 / dt ** 2.0) * (DTI[I] - DTIn[I] - (dDTIn[I] * dt))) - ddDTIn[I]

    # Coding first derivative of Temperature w.r.t to time
    dDTI   = np.zeros(4)

    for I in range(4):
        dDTI[I] = ((2 / dt) * (DTI[I] - DTIn[I])) - dDTIn[I]


    # write histroy - Newmark time integration
    [[Ht[0],  Ht[1] ], [Ht[2] , Ht[3] ], [Ht[4] , Ht[5] ], [Ht[6] , Ht[7]] ] = aI
    [[Ht[8],  Ht[9] ], [Ht[10], Ht[11]], [Ht[12], Ht[13]], [Ht[14], Ht[15]]] = vI
    [[Ht[16], Ht[17]], [Ht[18], Ht[19]], [Ht[20], Ht[21]], [Ht[22], Ht[23]]] = uI 
    [Ht[24+0], Hn[24+1], Ht[24+2 ], Ht[24+3 ]] = ddDTI
    [Ht[24+4], Hn[24+5], Ht[24+6 ], Ht[24+7 ]] = dDTI
    [Ht[24+8], Ht[24+9], Ht[24+10], Ht[24+11]] = DTI 
    
    # provide integration points
    aa = 1/np.sqrt(3)
    EGP = np.array([[-aa, -aa, 1],[aa, -aa, 1],[aa, aa, 1],[-aa, aa, 1]])
    NoInt = len(EGP)

    # start integration Loop
    for GP in range(NoInt):
        xi, eta, wgp  = EGP[GP]
        #if verbose: print('GP: ',GP,' Xi_gp = [',xi,', ',eta,' ]')


        # evaluate shape functions at this gp

        SHP = 1/4 * np.array([(1.0-xi)*(1.0-eta), (1.0+xi)*(1.0-eta), (1.0+xi)*(1.0+eta), (1.0-xi)*(1.0+eta)])

        SHP_dxi = 1/4 * np.array([  [ -(1.0-eta),  -(1.0-xi)],
                                    [  (1.0-eta),  -(1.0+xi)],
                                    [  (1.0+eta),   (1.0+xi)],
                                    [ -(1.0+eta),   (1.0-xi)]
                                 ], dtype=np.float64)

        acc = np.zeros(2)
        for i in range(2):
            for I in range(4):
                acc[i] += SHP[I] * aI[I, i]

        temp_rate = 0.0
        for I in range(4):
            temp_rate += SHP[I] * dDTI[I]

        # compute Jacobian matrix
        J = np.zeros((2, 2))
        for I in range(4):
            for i in range(2):
                for j in range(2):
                    J[i, j] += SHP_dxi[I, j] * xI[I, i]

        # compute Jacobi- determinant and inverse
        detJ = np.linalg.det(J)
        Jinv = np.linalg.inv(J)

        # compute gradient shape functions
        SHP_dx = np.zeros((4, 2))
        for I in range(4):
            for i in range(2):
                for j in range(2):
                    SHP_dx[I, i] += Jinv[j, i] * SHP_dxi[I, j]

        # compute strains
        eps = np.zeros(6)
        for I in range(4):
            # compute B-matrix for this node I
            BI = np.array([ [SHP_dx[I,0], 0           ],
                            [0          , SHP_dx[I,1] ],
                            [0          , 0           ],
                            [SHP_dx[I,1], SHP_dx[I,0] ],
                            [0          , 0           ],
                            [0          , 0           ]
                        ])
            for i in range(6):
                for j in range(2):
                    eps[i] += BI[i, j] * uI[I, j]

        # compute strain rates
        epsrt = np.zeros(6)
        for I in range(4):
            # compute B-matrix for this node I
            BI = np.array([[SHP_dx[I, 0],            0],
                           [0,            SHP_dx[I, 1]],
                           [0,                       0],
                           [SHP_dx[I, 1], SHP_dx[I, 0]],
                           [0,                       0],
                           [0,                       0]
                          ])
            for i in range(6):
                for j in range(2):
                    epsrt[i] += BI[i, j] * vI[I, j]


        # Computing the Gaussian Temperature
        Gauss_temp = 0.0
        for I in range(4):

            # select shape function at node I
            NI = SHP[I]
            Gauss_temp += SHP[I]*DTI[I]


        # form constitutive tensor
        lam, mue = (Emod*nu)/((1.0+nu)*(1.0-2.0*nu)), Emod/(2.0*(1.0+nu))
        Cmat = np.array([
                [lam + 2* mue, lam         , lam         , 0  , 0  , 0  ],
                [lam         , lam + 2* mue, lam         , 0  , 0  , 0  ],
                [lam         , lam         , lam + 2* mue, 0  , 0  , 0  ],
                [0           , 0           , 0           , mue, 0  , 0  ],
                [0           , 0           , 0           , 0  , mue, 0  ],
                [0           , 0           , 0           , 0  , 0  , mue]
                ], dtype=np.float64)

        Dmat = np.array([
                [mu_eta, 0, 0, 0, 0, 0],
                [0, mu_eta, 0, 0, 0, 0],
                [0, 0, mu_eta, 0, 0, 0],
                [0, 0, 0, 0.5*mu_eta, 0, 0],
                [0, 0, 0, 0, 0.5*mu_eta, 0],
                [0, 0, 0, 0, 0, 0.5*mu_eta]
                ], dtype=np.float64)

        # Identity Matrix voigt notation
        Iden = np.zeros(6)
        Iden[0] = 1
        Iden[1] = 1
        Iden[2] = 1
        Iden[3] = 0
        Iden[4] = 0
        Iden[5] = 0

        # compute stresses

        sig = np.zeros(6)
        for i in range(6):
            sig[i] += -3*a_T*kappa*Gauss_temp*Iden[i]
            for m in range(6):
                sig[i] += Cmat[i, m] * eps[m] + Dmat[i, m] * epsrt[m]

        # Computing temperature gradient

        gradtheta = np.zeros(2)
        BK = np.zeros(2)
        for I in range(4):
            # compute B-matrix for this node I
            for i in range(2):
                BK[i] = SHP_dx[I,i]
            for i in range(2):

                gradtheta[i] += BK[i]*DTI[I]






        BJ = np.zeros(2)
        BN = np.zeros(2)
        # compute element vector and matrix
        for I in range(4):

            # select shape function at node I
            NI = SHP[I]
            # compute B-matrix for this node I
            BX = np.array([[SHP_dx[I, 0],            0],
                           [0,            SHP_dx[I, 1]],
                           [0,                       0],
                           [SHP_dx[I, 1], SHP_dx[I, 0]],
                           [0,                       0],
                           [0,                       0]
                           ])

            for i in range(2):
                BJ[i] = SHP_dx[I, i]
            

            for k in range(2):
                r_e[(I * 3) + k] += (rho * (acc[k] - b[k]) * NI) * detJ * wgp
                for i in range(6):
                    r_e[(I * 3) + k] += (sig[i] * BX[i, k]) * detJ * wgp

            

            r_e[I*3+2] += (- rho * r * SHP[I]) * detJ * wgp  + (rho * c * temp_rate * SHP[I]) * detJ * wgp
            for i in range(2):
                r_e[I*3+2] += (a_q*gradtheta[i]*BJ[i])*detJ*wgp

            for J in range(4):
                # select shape function at node J
                NJ = SHP[J]
                # compute B-matrix for this node J
                BM = np.array([[SHP_dx[J, 0], 0],
                               [0, SHP_dx[J, 1]],
                               [0, 0],
                               [SHP_dx[J, 1], SHP_dx[J, 0]],
                               [0, 0],
                               [0, 0]
                               ])

                for i in range(2):
                    BN[i] = SHP_dx[J, i]


                for k in range(2):
                    for l in range(2):
                        if k == l:
                            alpha = 1
                        else:
                            alpha = 0
                        k_e[I*3+k, J*3+l] += ((4 * rho / dt ** 2) * NI * NJ * alpha)* detJ * wgp
                        for i in range(6):
                            for m in range(6):

                                k_e[I*3+k, J*3+l] += Cmat[i, m] * BM[m,l] * BX[i,k] * detJ * wgp


                k_e[3*I+2, 3*J+2] += (rho * c * NJ * NI * 2 / dt) * detJ * wgp
                for i in range(2):
                    k_e[3*I+2, 3*J+2] += (a_q*BJ[i]*BN[i])*detJ*wgp


                for m in range(6):
                    k_e[I*3, 3*J+2] += (-3*a_T*kappa*NJ*Iden[m]*BX[m, 0])*detJ*wgp
                    k_e[(I*3)+1, 3*J+2] += (-3*a_T*kappa*NJ*Iden[m]*BX[m, 1])*detJ*wgp



    return r_e, k_e


def Elmt_Post(XL, UL, Hn, Ht, Mat, dt, PostName):
    '''
    The Elmt_Post(XL, UL, Hn, Ht, Mat, dt, PostName) function returns a vector,
    containing a scalar for each node.
    '''

    # initialize return - the post function always returns a vector 
    # containing one scalar per node of the element
    r_post = np.zeros(4)

    # read in material parameter
    Emod, nu, rho, mu_eta, bx, by, a_q, c, r, T0, a_T = Mat
    b = np.array([bx, by])

    # restructure input to fit our notation
    xI = np.array([[XL[0], XL[1]], [XL[2], XL[3]], [XL[4], XL[5]], [XL[6], XL[7]]])
    uI = np.array([[UL[0], UL[1]], [UL[3], UL[4]], [UL[6], UL[7]], [UL[9], UL[10]]])
    DTI = np.array([UL[2], UL[5], UL[8], UL[11]])

    # read histroy - Newmark time integration
    aIn    = np.array([[Hn[0],  Hn[1] ], [Hn[2] , Hn[3] ], [Hn[4] , Hn[5] ], [Hn[6] , Hn[7]] ])
    vIn    = np.array([[Hn[8],  Hn[9] ], [Hn[10], Hn[11]], [Hn[12], Hn[13]], [Hn[14], Hn[15]]])
    uIn    = np.array([[Hn[16], Hn[17]], [Hn[18], Hn[19]], [Hn[20], Hn[21]], [Hn[22], Hn[23]]])
    ddDTIn = np.array([Hn[24+0], Hn[24+1], Hn[24+2 ], Hn[24+3 ]])
    dDTIn  = np.array([Hn[24+4], Hn[24+5], Hn[24+6 ], Hn[24+7 ]])
    DTIn   = np.array([Hn[24+8], Hn[24+9], Hn[24+10], Hn[24+11]])

    # compute current acceleration and velocity - Newmark time integration
    aI     = np.zeros((4,2))
    vI     = np.zeros((4,2))
    ddDTI  = np.zeros(4)
    dDTI   = np.zeros(4)

    # write histroy - Newmark time integration
    [[Ht[0],  Ht[1] ], [Ht[2] , Ht[3] ], [Ht[4] , Ht[5] ], [Ht[6] , Ht[7]] ] = aI
    [[Ht[8],  Ht[9] ], [Ht[10], Ht[11]], [Ht[12], Ht[13]], [Ht[14], Ht[15]]] = vI
    [[Ht[16], Ht[17]], [Ht[18], Ht[19]], [Ht[20], Ht[21]], [Ht[22], Ht[23]]] = uI 
    [Ht[24+0], Hn[24+1], Ht[24+2 ], Ht[24+3 ]] = ddDTI
    [Ht[24+4], Hn[24+5], Ht[24+6 ], Ht[24+7 ]] = dDTI
    [Ht[24+8], Ht[24+9], Ht[24+10], Ht[24+11]] = DTI 
    
    # provide integration points
    aa = 1/np.sqrt(3)
    EGP = np.array([[-aa, -aa, 1],[aa, -aa, 1],[aa, aa, 1],[-aa, aa, 1]])
    NoInt = len(EGP)

    #computing bulk modulus

    kappa = 0.0

    kappa = Emod / (3*(1-(2*nu)))


    # start integration Loop
    for GP in range(NoInt):
        xi, eta, wgp  = EGP[GP]

        # evaluate shape functions at this gp

        SHP = 1 / 4 * np.array(
            [(1.0 - xi) * (1.0 - eta), (1.0 + xi) * (1.0 - eta), (1.0 + xi) * (1.0 + eta), (1.0 - xi) * (1.0 + eta)])

        SHP_dxi = 1 / 4 * np.array([[-(1.0 - eta), -(1.0 - xi)],
                                    [(1.0 - eta), -(1.0 + xi)],
                                    [(1.0 + eta), (1.0 + xi)],
                                    [-(1.0 + eta), (1.0 - xi)]
                                    ], dtype=np.float64)

        # compute Jacobian matrix
        J = np.zeros((2, 2))
        for I in range(4):
            for i in range(2):
                for j in range(2):
                    J[i, j] += SHP_dxi[I, j] * xI[I, i]

        # compute Jacobi- determinant and inverse
        detJ = np.linalg.det(J)
        Jinv = np.linalg.inv(J)

        # compute gradient shape functions
        SHP_dx = np.zeros((4, 2))
        for I in range(4):
            for i in range(2):
                for j in range(2):
                    SHP_dx[I, i] += Jinv[j, i] * SHP_dxi[I, j]

        # compute strains
        eps = np.zeros(6)
        for I in range(4):
            # compute B-matrix for this node I
            BI = np.array([[SHP_dx[I, 0], 0],
                           [0, SHP_dx[I, 1]],
                           [0, 0],
                           [SHP_dx[I, 1], SHP_dx[I, 0]],
                           [0, 0],
                           [0, 0]
                           ])
            for i in range(6):
                for j in range(2):
                    eps[i] += BI[i, j] * uI[I, j]

        # compute strain rates
        epsrt = np.zeros(6)
        for I in range(4):
            # compute B-matrix for this node I
            BI = np.array([[SHP_dx[I, 0], 0],
                           [0, SHP_dx[I, 1]],
                           [0, 0],
                           [SHP_dx[I, 1], SHP_dx[I, 0]],
                           [0, 0],
                           [0, 0]
                           ])
            for i in range(6):
                for j in range(2):
                    epsrt[i] += BI[i, j] * vI[I, j]

        # Computing the Gaussian Temperature
        Gauss_temp = 0.0
        for I in range(4):
            # select shape function at node I
            NI = SHP[I]
            Gauss_temp += SHP[I] * DTI[I]

        # form constitutive tensor
        lam, mue = (Emod * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu)), Emod / (2.0 * (1.0 + nu))
        Cmat = np.array([
            [lam + 2 * mue, lam, lam, 0, 0, 0],
            [lam, lam + 2 * mue, lam, 0, 0, 0],
            [lam, lam, lam + 2 * mue, 0, 0, 0],
            [0, 0, 0, mue, 0, 0],
            [0, 0, 0, 0, mue, 0],
            [0, 0, 0, 0, 0, mue]
        ], dtype=np.float64)

        Dmat = np.array([
            [mu_eta, 0, 0, 0, 0, 0],
            [0, mu_eta, 0, 0, 0, 0],
            [0, 0, mu_eta, 0, 0, 0],
            [0, 0, 0, 0.5 * mu_eta, 0, 0],
            [0, 0, 0, 0, 0.5 * mu_eta, 0],
            [0, 0, 0, 0, 0, 0.5 * mu_eta]
        ], dtype=np.float64)

        # Identity Matrix voigt notation
        Iden = np.zeros(6)
        Iden[0] = 1
        Iden[1] = 1
        Iden[2] = 1
        Iden[3] = 0
        Iden[4] = 0
        Iden[5] = 0

        # compute stresses

        sig = np.zeros(6)
        for i in range(6):
            sig[i] += -3 * a_T * kappa * Gauss_temp * Iden[i]
            for m in range(6):
                sig[i] += Cmat[i, m] * eps[m] + Dmat[i, m] * epsrt[m]

        # compute vonMises stresses
        sig_vm = (0.5 * ((sig[0] - sig[1]) ** 2 + (sig[0] - sig[2]) ** 2 + (sig[1] - sig[2]) ** 2) + 3.0 * (
                            sig[3] ** 2 + sig[4] ** 2 + sig[5] ** 2)) ** (0.5)

        if (PostName=="SigMises"):
            r_post += sig_vm * SHP

    # based on the string PostName different output is returned
    if (PostName=="UX"):
        r_post = np.array([UL[0], UL[3], UL[6], UL[9]])
        return r_post
    elif (PostName=="UY"):
        r_post = np.array([UL[1], UL[4], UL[7], UL[10]])
        return r_post
    elif (PostName=="T"):
        r_post = np.array([T0+UL[2], T0+UL[5], T0+UL[8], T0+UL[11]])
        return r_post
    elif (PostName=="SigMises"):
        return r_post
    else:
        print("Waring: PostName "+PostName+" not defined!")
        return np.array([0.0, 0.0, 0.0, 0.0])


## This is a sanity check for the element

# define dummy input
XL = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
UL = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0])
Hn = np.zeros(4*3*3)
Ht = np.zeros(4*3*3)
Mat = [2100, 0.3, 1.0, 3.0, 0.1, 2.0, 10, 5.0, 0, 293.15, 10e-5]
dt = 1
# call the elemnt with this dummy input
re, ke = Elmt_KS(XL, UL, Hn, Ht, Mat, dt)
# check the resulting vector / matrix
print('r_e :')
print(re)
print('k_e :')
print(ke)
