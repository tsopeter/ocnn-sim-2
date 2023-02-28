function z = sa_backward(E)
    a0 = 20;
    g  = E;

    m = exp(-a0/2/(1+g^2));
    z = single(m * (1+(a0*g*g)/(1+g^2)^2));
end