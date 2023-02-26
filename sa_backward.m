function z = sa_backward(E)
    a0 = 20;
    g  = E;

    if (g < 0.0001)
        m = 0.0001;
        z = m;
    else
        q = abs(sa_forward(E))/g;
        m = q * exp(1+(a0*g*g)/(1+g^2)^2);
        z = m;
    end
end