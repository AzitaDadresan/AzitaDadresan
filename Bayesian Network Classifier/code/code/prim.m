% Code Reference: https://stackoverflow.com/questions/53568255/find-a-minimum-spanning-tree
function mst = prim(A)

  % User supplies adjacency matrix A.  This program uses Prims algorithm
  % to find a minimum spanning tree.  The edges of the minimum spanning
  % tree are returned in array mst (of size n-1 by 2)

  [n,n] = size(A);                           % The matrix is n by n, where n = # nodes.

  if (norm(A-transpose(A),'fro') ~= 0)       % If adjacency matrix is not symmetric,
    disp(' Error:  Adjacency matrix must be symmetric ') % print error message and quit.
    return;
  end

  % Start with node 1 and keep track of which nodes are in tree and which are not.
  intree = [1];    
  notintree = [2:n]; 
  number_of_edges = 0;

  % Iterate until all n nodes are in tree.

  while numel(intree) < n
    % Find the maximum edge from a node that is in tree to one that is not.
    maxcost = -Inf;                             % You can actually enter infinity into Matlab.
    for i=1:numel(intree)
      for j=1:numel(notintree)
        ii = intree(i);  
        jj = notintree(j);
        if A(ii,jj) > maxcost 
          maxcost = A(ii,jj); 
          jsave = j;
          iisave = ii; 
          jjsave = jj;  % Save coords of node.
        end
      end
    end

    % Add this edge and associated node jjsave to tree.  
    % Delete node jsave from list of those not in tree.

    number_of_edges = number_of_edges + 1;      % Increment number of edges in tree.
    mst(number_of_edges,1) = iisave;            % Add this edge to tree.
    mst(number_of_edges,2) = jjsave;

    intree = [intree; jjsave];                  % Add this node to tree.
    notintree(jsave) = [];                          % Delete this node from list of those not in tree.

end;