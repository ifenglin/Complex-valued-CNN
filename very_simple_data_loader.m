function [ city, field, forest, grass, street ] = very_simple_data_loader( size, ch, num )
    % create very simple data 
    city = complex(0, rand(size, size, ch, num));
    field = complex(rand(size, size, ch, num), rand(size, size, ch, num)* 1e-1) ;
    forest = complex(rand(size, size, ch, num), rand(size, size, ch, num))* 1e-1 ;
    grass = complex(rand(size, size, ch, num), rand(size, size, ch, num));
    street = complex(rand(size, size, ch, num), 0);
end

