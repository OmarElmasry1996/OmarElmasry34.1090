function [ deNormalisedTest ] = denormaliseSTD( hFinal,mu,sDeviation )


deNormalisedTest = zeros(size(hFinal));               %normailising

                          
    deNormalisedTest = (hFinal*sDeviation) + mu;

end

