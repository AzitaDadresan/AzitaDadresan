function [prior, P_X, P_XY, mst] = train_Chow_Liu(train_data, train_labels, fmax, nclass)
    % train_data should be of dimension m x n where m is the number of training samples and n is the number of features
    % The values in train data should be positive integers corresponding to different values of features (1,2 etc.)
    % train_labels should be of dimension m x 1. It contains the class label of each training sample (1,2 etc.)
    % fmax should be of dimension 1 x n. It should indicate the maximum value each feature can take
    % nclass is the number of classes. % nclass = 2 for binary classification.
    % Returns probability values and maximum spanning trees learned by the Chow Liu Algorithm

    [m, n] = size(train_data); % n is number of features

    % Prior probability 
    prior = zeros(nclass, 1); % a vector of dimension nclass

    % probability of a feature taking a particular value
    P_X = zeros(nclass, n, max(fmax)); 

    % probability of two features X & Y taking a particular pair of values
    P_XY = zeros(nclass, n, n, max(fmax), max(fmax)); 

    % mst: maximum spanning trees; dimension nclass x (n-1) x 2
    mst = zeros(nclass, n-1, 2);

    for a_class = 1:nclass

        % train data corresponding to a class
        idx = (train_labels == a_class);
        train_data_c = train_data(idx, :);

        % number of samples belonging to the class 
        nC = sum(idx);

        % prior probability of the class
        prior(a_class) = nC / m;


        % Count how many times each feature takes a value  
        Count_X = zeros(n, max(fmax));

        for X = 1:n
            for value = 1:fmax(X)
                Count_X(X, value) = sum(train_data_c(:, X) == value);
            end
        end

        % Compute probability values using MLE (maximum likelihood estimation)
        % We also perform add-one-smoothing to avoid zero probability values.
        P_X(a_class, :, :) = (Count_X + 1) ./ transpose(fmax + nC);


        % Count how many times two features X & Y come together 
        % There are n*n pairs of XY and fmax(X) x fmax(Y) pairs of possible values for each XY pair
        Count_XY = zeros(n, n, max(fmax), max(fmax)); % A pair XY can take 4 values (1,1), (1,2), (2,1) and (2,2) usually
        pairs_max = zeros(n, n); % how many pairs of values in each XY

        % Count how many times two features X & Y come together in Class a_class       
        for X = 1:n-1
            for Y = X+1:n
                % count how many times X&Y takes each pair of possible values
                for valx = 1:fmax(X)
                    for valy = 1:fmax(Y)
                        X1 = train_data_c(:, X) == valx; % boolean vectors indicating feature X is valx
                        Y1 = train_data_c(:, Y) == valy; % boolean vectors indicating feature Y is valy
               
                        % Take boolean AND of X1 and Y1 to get the situation where X=valx and Y=valy
                        X1Y1 = sum(X1 .* Y1); % Count how many times X=valx and Y=valy
                        Count_XY(X, Y, valx, valy) = X1Y1;
                    end
                end

                pairs_max(X, Y) = fmax(X) * fmax(Y);
            end
        end

        % perform add-one-smoothing to avoid zero probability values and hence to avoid log(0)
        P_XY(a_class, :, :, :, :) = (Count_XY + 1) ./ (pairs_max + nC); 

        % Compute the Mutual Information Matrix I(X, Y) 
        I = zeros(nclass, n, n);
        
        for X = 1:n-1
            for Y = X+1:n
                % mutual information for a class (refer equation)
                mi = 0;
                for valx = 1:fmax(X)
                    for valy = 1:fmax(Y)
                        mi = mi + P_XY(a_class, X, Y, valx, valy) ...
                                * log(P_XY(a_class, X, Y, valx, valy)/(P_X(a_class, X, valx)*P_X(a_class, Y, valy)));
                    end
                end

                I(a_class, X, Y) = mi;
                I(a_class, Y, X) = mi;
            end
        end

        % Find the MST (Maximum Spanning Tree) for each class
        T = prim(squeeze(I(a_class, :, :)));
        mst(a_class, :, :) = T;
    end
end
