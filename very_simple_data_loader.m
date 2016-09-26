function [ city, field, forest, grass, street ] = very_simple_data_loader( size, ch, num )
    % create very simple data 
    city = complex(rand(size, size, ch, num), 10+rand(size, size, ch, num));
    field = complex(10+rand(size, size, ch, num), 10+rand(size, size, ch, num)) ;
    forest = complex(10+rand(size, size, ch, num), -10+rand(size, size, ch, num));
    grass = complex(-10+rand(size, size, ch, num), 10+rand(size, size, ch, num));
    street = complex(-10+rand(size, size, ch, num), -10+rand(size, size, ch, num));
end

