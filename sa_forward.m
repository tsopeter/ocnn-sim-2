function z = sa_forward(E)
    a0 = 20;
    g = E;
    z = exp(-a0/2/(1+g^2))*g;


end