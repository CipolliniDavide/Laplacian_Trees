/* PURD/1NN Portegys/Mangalagiu/Schomaker */

/*
    #define WSD         for an experiment with a distance measure where feature weighing is used with 1/SD  (inverse of
        standard deviation)

    #define KULLBACK    for an experiment using Kullback-Leibler relative information These two experiments did not 
                        yield substantial improvements.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#define TIMING_START 1
#define TIMING_STOP 2

#define MINDIST 0.0000001
#define MAXITER 10

#define DEBUGP 0
#define dprintf                                                                                                        \
    if (DEBUGP)                                                                                                        \
    printf
int NSAMP_IF_RANDOM = 500;

#define MATCH_1NN_ABS 1
#define MATCH_1NN_REL 2
#define MATCH_kNN_REL 3
#define MATCH_NCENT_1NN 4
#define MATCH_NCENT_FRQ 5
int MatchMethod = MATCH_1NN_ABS;

#define MAXKNN 1000

#define TEST_MODE 1
#define TYPER_MODE 2

#define INSERT_OK 12340
#define PROBLEM 12341
#define NEWNODE_EQUALS_PARENT 12342
#define NEWNODE_EQUALS_PARENT_ACCORDING_TO_DIST 12343
#define SIBBACK_EQUALS_PARENT_CHILDLAST 12344

int Nmin_NCENT = 2;
int K_KNN = 1;
int Kmin_KNN = 1;
int Kmax_KNN = MAXKNN;
int Ndist = 0;

double Wkb[256]; /* pixel vector */

typedef double Feature;

FILE *fpg = NULL;

typedef struct _NODE_ {
    struct _NODE_ *Childlist; /* Pointer to next downward child */
    struct _NODE_ *Childlast; /* Pointer to last child in linked list */
    struct _NODE_ *Sibnext;   /* Pointer to next rightward sibbling */
    struct _NODE_ *Sibback;   /* Pointer to start node of this siblist (jos: previous, not first according to the paper) */
    struct _NODE_ *myparent;  /* Pointer to the parent */
    double parentdist;        /* Distance to parent during construction */
    int Nchildren;            /* Number of children */
    Feature *Pattvec;         /* Pointer to the feature vector */
    Feature *Centvec;         /* Pointer to malloced centroid vector */
#ifdef WSD
    Feature *Centvsq; /* Pointer to malloced centroid vector */
#endif
    int CharFreq[128]; /* charclass histogram */
    int majority;      /* ASCII index of majority pattern centroid */
    int nfreq;         /* number of frequentations in node typing */
    int iclass;        /* ASCII number of character */
    int nmemb;         /* Number of vectors used in comp. centroid*/
    int Nfeat;         /* Number of features */
    int instance_id;   /* Index of the original feature vector */
    int had;           /* Traversal flag */
    int lev;           /* Node depth */
#define DIST_BOOKKEEPING
#ifdef DIST_BOOKKEEPING
    int dcalced;     /* flag whether dist() with probe is known */
    float probedist; /* distance of this node with probe */
#endif
} Node;

void traverse_making_centroids(Node *mother, Node *node, int lev, int logging);

void traverse_search(Node *node, int lev, int logging, double radius, double *vec, char *lab, int *iout, double *dout,
                     int *k_found, int mode);

void traverse_children(Node *node, int lev, int *ntraversed, int logging);

/* ............................................................... */

/* --- Jos: Some Utility Functions --- */
int does_folder_exist(const char *name) {
    struct stat sb;
    return stat(name, &sb) == 0 && S_ISDIR(sb.st_mode);
}

void create_folder(const char* name) {
    mkdir(name, 0700);
}

void save_node_class_mapping(char* filename, Node* nodes, int nobs) {
    FILE *fpt = fopen(filename, "w");
    if (fpt == NULL) {
        fprintf(stderr, "Error opening output file [%s]\n", filename);
        exit(1);
    }
    
    fprintf(fpt, "Index: Class Mapping\n");
    for (int index = 0; index < nobs; ++index) {
        fprintf(fpt, "%d: %d\n", index, nodes[index].iclass - '0');
    }
    fclose(fpt);
}

/*
    Finds the last child in the linked list
*/
Node *find_childlast(Node *p) {
    Node *a;
    a = NULL;
    if (p != NULL) {
        a = p;
        while (1) {
            printf("c");
            if (a->Childlist == NULL)
                break;
            a = a->Childlist;
        }
    }
    return (a);
}

/*
    Returns the node id, or -999 if the node is NULL
*/
int idNode(Node *p) {
    if (p != NULL) {
        return (p->instance_id);
    } else {
        return (-999);
    }
}

/*
    Starts or reads a timer, there are 10 timers available by index
*/
double timing(int mode, int nfrag, char *title, int itimer) {
    static time_t t0[10], t[10], dt;

    if (mode == TIMING_START) {
        t0[itimer] = clock();
        dt = 0.0;
    } else {
        t[itimer] = clock();
        dt = ((unsigned int)t[itimer] - (unsigned int)t0[itimer]) / nfrag;
        printf(" %s: time per unit %.1f [us] (n=%d)\n", title, (1000000. * (double)dt) / (double)CLOCKS_PER_SEC, nfrag);
    }
    return ((1000000. * (double)dt) / (double)CLOCKS_PER_SEC);
}

/*
    Counts and returns the amount of siblings after this node
*/
int get_nsibnext(Node *o, int logging) {
    int i;
    Node *p;
    i = 0;
    p = o;
    while (p->Sibnext != NULL) {
        if (logging)
            dprintf("Sibnext %04d -> %04d\n", o->instance_id, p->Sibnext->instance_id);
        ++i;
        p = p->Sibnext;
    }
    return (i);
}

/*
    Counts and returns the amount of siblings before this node
*/
int get_nsibback(Node *o, int logging) {
    int i;
    Node *p;
    i = 0;
    p = o;
    while (p->Sibback != NULL) {
        printf("b");
        if (logging)
            dprintf("Sibback %04d <- %04d\n", o->instance_id, p->Sibback->instance_id);
        if (p == p->Sibback) {
            printf(".. is a Sibback inconsistency: selflink\n");
            break;
        }
        ++i;
        p = p->Sibback;
    }
    return (i);
}

/*
    Counts and returns the amount of downward children of this node
*/
int get_nchildren(Node *o, int logging) {
    int i;
    Node *p;
    i = 0;
    p = o;
    while (p->Childlist != NULL) {
        if (logging)
            dprintf("%04d has-child %04d\n", o->instance_id, p->Childlist->instance_id);
        ++i;
        p = p->Childlist;
    }
    return (i);
}

/*

*/
double fanout_factor(Node *o, int Nobs) {
    double r;
    Node *child;

    child = o->Childlist;

    r = get_nsibnext(child, 0);
    return (r / (double)Nobs);
}

void save_node(FILE *fpt, Node *o, int inode) {
    fprintf(fpt, "Node %d lev %d %.2f nc %d c %d %d s %d %d\n", idNode(o), o->lev, o->parentdist, o->Nchildren,
            idNode(o->Childlist), idNode(o->Childlast), idNode(o->Sibnext), idNode(o->Sibback));
}

void save_tree(char *filename, Node *o, int Nnodes, Node *root) {
    int i;
    FILE *fpt;

    fpt = fopen(filename, "w");
    if (fpt == NULL) {
        fprintf(stderr, "Error opening outputfile [%s]\n", filename);
        exit(1);
    }

    fprintf(fpt, "n %d nfeat %d root %d\n", Nnodes, o[0].Nfeat, idNode(root));

    for (i = 0; i < Nnodes; ++i) {
        save_node(fpt, &o[i], i);
    }
    fclose(fpt);
}

void print_node(Node *o, int inode) {
    int nsibn, nsibb, nch, pid;

    nsibn = get_nsibnext(o, 1);
    nsibb = get_nsibback(o, 1);
    nch = get_nchildren(o, 1);

    pid = -1;
    if (o->myparent != NULL) {
        pid = o->myparent->instance_id;
    }

    printf("Node i=%d id=%04d pd=%f nsibnext=%d nsibback=%d nchild=%d "
           "parent=%04d\n",
           inode, o->instance_id, o->parentdist, nsibn, nsibb, nch, pid);
}

void save_hist(char *filename, Node *o, int Nnodes, Node *root) {
    int i, j, l, maxlev;
    FILE *fpt;

    fpt = fopen(filename, "w");
    if (fpt == NULL) {
        fprintf(stderr, "Error opening outputfile [%s]\n", filename);
        exit(1);
    }

    maxlev = 0;
    for (i = 0; i < Nnodes; ++i) {
        if (o[i].lev > maxlev) {
            maxlev = o[i].lev;
        }
    }

    fprintf(fpt, "%d samples\n", Nnodes);

    for (l = maxlev; l > 0; l--) {
        fprintf(fpt, "step %d:\n", maxlev - l);
        for (i = 0; i < Nnodes; ++i) {
            if (o[i].lev == l) {
                for (j = 0; j < Nnodes; ++j) {
                    if (idNode(o[i].myparent) == j) {
                        fprintf(fpt, "   %d -> %d\n", i, j);
                    }
                }
            }
        }
    }
    fclose(fpt);
}

// #define Kullback
#ifdef Kullback
double dist(Feature *vec1, Feature *vec2, int nf) {
    int i;
    double d, dsum;
    dsum = 0.0;
    for (i = 0; i < nf; ++i) {
        d = vec1[i] - vec2[i];
        dsum += d * d * Wkb[i];
    }
    ++Ndist;
    return (sqrt(dsum));
}
#else
double dist(Feature *vec1, Feature *vec2, int nf) {
    int i;
    double d, dsum;
    dsum = 0.0;
    for (i = 0; i < nf; ++i) {
        d = vec1[i] - vec2[i];
        dsum += d * d;
    }
    ++Ndist;
    return (sqrt(dsum));
}
#endif

#ifdef WSD
double distw(Feature *vec1, Node *p) {
    int i;
    double d, dsum;

    dsum = 0.0;
    for (i = 0; i < p->Nfeat; ++i) {
        d = (vec1[i] - p->Centvec[i]) / p->Centvsq[i];
        dsum += d * d;
    }
    return (sqrt(dsum));
}
#endif

double dist_match(Feature *S, Node *pB, double radius, int method) {
    int nf;
    double d, *B;

    nf = pB->Nfeat;

    B = pB->Pattvec;

    if (method == MATCH_1NN_ABS) {
        d = dist(S, B, nf);
    } else if (method == MATCH_1NN_REL) {
        d = dist(S, B, nf) - pB->parentdist * radius;
    } else if (method == MATCH_NCENT_1NN || method == MATCH_NCENT_FRQ) {
        d = dist(S, pB->Centvec, nf);
#ifdef WSD
    } else if (method == MATCH_WCENT) {
        d = distw(S, pB);
#endif
    } else {
        fprintf(stderr, "MatchMethod %d not implemented yet\n", method);
        exit(1);
    }
    return (d);
}

/*
    This function implements the insertion algorithm as described in Portegys his article.
*/
int insert_pattern_portegys(Node *new_Node, Node *Parent, double radius, int Nobs, int *visited) {
    Node *OrigParent, *Child, *Temp1, *Temp2, *Siblist, *Childlist, *Childlast;
    int nc, flip;
    static int istat;

    *visited += 1;
    if (*visited > (Nobs * 10)) {
        return (PROBLEM);
    }

    /* Calculate the distance between the new node and the current parent */
    new_Node->parentdist = dist(new_Node->Pattvec, Parent->Pattvec, Parent->Nfeat);

    dprintf("\ninsport new=%04d(pd=%f) vs parent=%04d(pd=%f)\n", idNode(new_Node), new_Node->parentdist, idNode(Parent),
            Parent->parentdist);

    if (new_Node == Parent) {
        dprintf("insert_pattern_portegys() identical nodes\n");
        return (NEWNODE_EQUALS_PARENT);
    }

    new_Node->myparent = Parent;

    // Check whether the distance is not 0 (disregarding floating point rounding errors)
    // if it is it means that the parent and new node are equal
    if (new_Node->parentdist < MINDIST) {
        dprintf("If dist(new_Node=%04d,Parent=%04d) < %f then new_Node := Parent\n", idNode(new_Node), idNode(Parent),
                MINDIST);
        /* new_Node == Parent; */
        return (NEWNODE_EQUALS_PARENT_ACCORDING_TO_DIST);
    } else {
        // if they are not equal however:
        dprintf("find child, then its siblings which has to be replaced\n");
        flip = 0;

        // This loop places the new node as deep in the tree as possible / required
        while (1) {
            nc = 0;

            OrigParent = Parent;

            // Iterate over all parent's direct children (1 step down)
            for (Child = Parent->Childlist; Child != NULL; Child = Child->Sibnext) {
                ++nc;

                // if we the distance of the new_node to the child is less than between dist(parent_node, child) * radius
                // we consider the child to be the new parent. Specifically we match with the first child that matches
                if (dist(new_Node->Pattvec, Child->Pattvec, Child->Nfeat) <= Child->parentdist * radius) {
                    dprintf("%d search down from Parent=%04d with Child=%04d\n", nc, idNode(Parent), idNode(Child));
                    // We store the new found parent and break from this loop.
                    Parent = Child;
                    flip = 1;
                    // this puts us back in the while loop, and we will now check whether the new parent has siblings 
                    // that the new node is a match for
                    break;
                }
            }

            if (nc == 0) {
                dprintf("Parent %04d has no children \n", idNode(OrigParent));
            } else {
                dprintf("Nchildren=%d of Parent=%04d checked for replacement\n", nc, idNode(OrigParent));
            }

            // This breaks the outer loop
            if (Child == NULL)
                break;
        } /* end while */

        if (flip) {
            dprintf("Update new_Node %04d, becoming a new child of deeper "
                    "Parent %04d\n",
                    idNode(new_Node), idNode(Parent));
        } else {
            dprintf("Update new_Node %04d, becoming a new child of original "
                    "Parent %04d\n",
                    idNode(new_Node), idNode(Parent));
        }

        // We update the new parent of the new node to whatever we matched in the previous loop. If nothing matched
        // it will still be the original parent
        new_Node->myparent = Parent;
        
        // We assume that this new node has no siblings yet, by default. Note that is also holds for recursive calls
        // overwriting possible node content
        new_Node->Sibnext = NULL;
        dprintf("this new_Node has ->Childlist=%04d\n", idNode(new_Node->Childlist));

        // If the new node is not in the most recently added child of the parent it is a sibling of that
        // child (?) not sure here (Is this ever true? it should always be considering the other case is an error.)
        if (new_Node != Parent->Childlast) {
            new_Node->Sibback = Parent->Childlast;
        } else {
            // If we end up here it means the new_node is placed as a child of the parent
            Parent->Childlast = find_childlast(Parent);
            if (new_Node == Parent->Childlast) {
                Parent->Childlast = NULL;
                printf("Self %04d Sibback unrecoverable\n", idNode(new_Node));
            } else {
                printf("Self %04d Sibback link just avoided, found %04d\n", idNode(new_Node), idNode(new_Node->Sibback));
            }
            new_Node->Sibback = Parent->Childlast;
            return (SIBBACK_EQUALS_PARENT_CHILDLAST);
        }

        if (Parent->Childlast != NULL) {
            dprintf("Parent(=%04d)->Childlast(=%04d)->Sibnext(%04d) := "
                    "new_Node = %04d\n",
                    idNode(Parent), idNode(Parent->Childlast), idNode(Parent->Childlast->Sibnext), idNode(new_Node));

            Parent->Childlast->Sibnext = new_Node; /* attach as sibling to the end
                                                      of the child list of Parent */
        } else {
            dprintf("new_Node %04d becomes first child of Parent %04d\n", idNode(new_Node), idNode(Parent));

            // this should not occur, considering this is false by merit of the previous if-statement (of which we are 
            // in the else branch)
            if (Parent->Childlist != NULL) { 
                dprintf("overwriting(I) Parent(=%04d)->Childlist = %04d with "
                        "%04d (Childlast should not have been NULL??\n",
                        idNode(Parent), idNode(Parent->Childlist), idNode(new_Node));
            }
            // attach as first child to Parent
            Parent->Childlist = new_Node;
        }

        // we overwrite the ChildLast with the new node
        Parent->Childlast = new_Node;
        Parent->Nchildren++;

        /* checked until here: OK */

        Child = Parent->Childlist;

        if (Child != NULL) {
            dprintf("make sure that Parent=%04d children are copied to "
                    "new_Node=%04d...\n",
                    idNode(Parent), idNode(new_Node));
        }

        // For each child we need to check whether they fit to the newly inserted node, or stay on the parent.
        while (Child != NULL) {
            if (Child == new_Node) {
                dprintf("Child == new_Node=%04d, breaking list, next would "
                        "have been %04d\n",
                        idNode(new_Node), idNode(Child->Childlist));
                break; /* ok: the newnode is guaranteed to be the last one */

            } else {
                // if the new_Node is closer to this child than to the parent: Reduce the number of parent children
                // and add new_Node to Child
                if (dist(new_Node->Pattvec, Child->Pattvec, Child->Nfeat) <= ((new_Node->parentdist) * radius)) {
                    Parent->Nchildren--;
                    
                    Temp1 = Child->Sibnext; /* obtain first sibling of Child */

                    if (Child->Sibback != NULL) {
                        dprintf("Child=%04d is not the first in the list \n", idNode(Child));
                        // we set the new next node of the previous node, to the next node
                        Child->Sibback->Sibnext = Temp1;
                    } else {
                        dprintf("Child=%04d is the first in the list \n", idNode(Child));
                        if (Parent->Childlist != NULL) {
                            dprintf("overwriting(II)?? Parent(=%04d)->Childlist = %04d\n", idNode(Parent), idNode(Parent->Childlist));
                            if (Temp1 != NULL) {
                                dprintf("         with\n");
                                dprintf("              Parent->Childlist = %04d\n", idNode(Temp1));
                            }
                        }
                        // This is the first child in the list (i.e. none before)
                        // so we set the new head of the list to the next child
                        Parent->Childlist = Temp1;
                    }

                    // Set the new previous of the next to the previous of the child, effectively removing the current 
                    // Child from the linked list
                    if (Temp1 != NULL) {
                        Temp1->Sibback = Child->Sibback;
                    }

                    dprintf("Convert child=%04d subtree to sibling list\n", idNode(Child));

                    Siblist = Child->Sibnext;
                    Childlist = Child->Childlist;
                    Childlast = Child->Childlast;

                    /* means Temp1 and Temp2->Sibnext are set to NULL */
                    // The node has now completely been severed, so we clear all links
                    Child->Sibnext = NULL;
                    Child->Childlist = NULL;
                    Child->Childlast = NULL;
                    Child->Nchildren = 0;

                    Temp1 = Temp2 = Child;
                    while (Temp1 != NULL && Temp2 != NULL) {
                        dprintf("Merge: Child %04d siblings, %04d children\n", idNode(Temp1), idNode(Temp2));

                        // Here we move Temp1 to be the last sibling of Childs previous siblings
                        while (Siblist != NULL) {
                            if (Temp1->Sibnext != NULL) {
                                Temp1 = Temp1->Sibnext;
                            }
                            // dprintf("Temp1=%04d\n", idNode(Temp1));
                            Siblist = Temp1->Sibnext;
                        }
                        /* Temp1 is the last sibnext in the chain now */

                        // We look for the first Sibling with children (I think), and store it in Temp2
                        while (Temp2 != NULL && Childlist == NULL) {
                            Childlist = Temp2->Childlist;
                            Temp2 = Temp2->Sibnext;
                            // dprintf("Temp2=%04d\n", idNode(Temp2));
                        }

                        if (Temp2 == NULL) {
                            break;
                        }

                        Temp1->Sibnext = Childlist;
                        Temp2 = Temp2->Sibnext;
                    }

                    dprintf("Recurse: Insert children list (first=%04d) to "
                            "Parent=%04d \n",
                            idNode(Child), idNode(Parent));

                    Temp1 = Child;
                    while (Temp1 != NULL) {
                        Temp2 = Temp1->Sibnext;
                        istat = insert_pattern_portegys(Temp1, Parent, radius, Nobs, visited);
                        printf("istat=%d in_insert_pattern_portegys();\n", istat);
                        if (istat == PROBLEM) {
                            return (PROBLEM);
                        }
                        Temp1 = Temp2;
                    }

                    Child = Parent->Childlist;

                    if (Child != NULL) {
                        printf("\nRetest, starting with child=%04d of "
                               "parent=%04d #visited=%d\n",
                               idNode(Child), idNode(Parent), *visited);
                    }

                } else {
                    Child = Child->Sibnext;
                }
            }
        } /* end while */

        return (INSERT_OK);
    }
}

int add_vector_noise(FILE *fp, char ***Labels, Feature ***Observations, int Nobs, int *Nfeat, int logging) {
    int last, i;

    last = Nobs;

    *Nfeat = 1;
    (*Observations)[last] = (Feature *)malloc(*Nfeat * sizeof(Feature));
    if ((*Observations)[last] == NULL) {
        fprintf(stderr, "Error mallocing feature vector values\n");
        exit(1);
    }

    if (last == 0) {
        fprintf(stderr, "(Generating artificial data: random noise)\n");
    }
    (*Labels)[last] = (char *)malloc(15 * sizeof(char));
    if ((*Labels)[last] == NULL) {
        fprintf(stderr, "Error mallocing label list entry\n");
        exit(1);
    }

    sprintf((*Labels)[last], "%04d", last);
    if (logging == 2)
        fprintf(stdout, "%s ", (*Labels)[last]);

    for (i = 0; i < *Nfeat; ++i) {
        (*Observations)[last][i] = (double)drand48();
        if (logging == 2) {
            fprintf(stdout, "%f ", (*Observations)[last][i]);
        }
    }
    if (logging == 2)
        fprintf(stdout, "\n");

    if (last < NSAMP_IF_RANDOM) {
        return (1);
    } else {
        return (0);
    }
}

int add_vector_file(FILE *fp, char ***Labels, Feature ***Observations, int Nobs, int *Nfeat, int logging) {
    int last, i, nsamp, nchr;
    char named[40];
    char data[40];
    char label[256];

    last = Nobs;

    // if last == 0, i.e. no nodes have been stored yet read parameters from the top of the file
    if (last == 0) {
        if (fscanf(fp, "%s%s%d%d", named, data, &nsamp, Nfeat) != 4) {
            return (0);
        }
        fprintf(stdout, "%s %s %d %d\n", named, data, nsamp, *Nfeat);
        if (strcmp("NAMED", named) != 0 || strcmp("DATA", data) != 0) {
            fprintf(stderr, "Expecting NAMED DATA input file\n");
            exit(1);
        }
    }

    // allocates space for the features in the new observation
    (*Observations)[last] = (Feature *)malloc(*Nfeat * sizeof(Feature));
    if ((*Observations)[last] == NULL) {
        fprintf(stderr, "Error mallocing feature vector values\n");
        exit(1);
    }

    if (fscanf(fp, "%s", label) == EOF) {
        return (0);
    }

    // allocate space for the label, and copy it into the storage
    nchr = strlen(label) + 1;
    (*Labels)[last] = (char *)malloc(nchr * sizeof(char));
    if ((*Labels)[last] == NULL) {
        fprintf(stderr, "Error mallocing label list entry\n");
        exit(1);
    }
    strcpy((*Labels)[last], label);

    if (logging == 2)
        fprintf(stdout, "%s ", (*Labels)[last]);
    
    // write the features to the newly allocated feature array from file
    for (i = 0; i < *Nfeat; ++i) {
        fscanf(fp, "%lf", &(*Observations)[last][i]);
        if (logging == 2) {
            fprintf(stdout, "%f ", (*Observations)[last][i]);
        }
    }
    if (logging == 2)
        fprintf(stdout, "\n");

    return (1);
}

int add_vector(FILE *fp, char ***Labels, Feature ***Observations, int Nobs, int *Nfeat, int logging) {
    int last;

    last = Nobs;
    // last == 0 indicates that we are reading the file header, so we skip allocation space for features & labels.
    if (last > 0) {
        *Observations = (Feature **)realloc(*Observations, (last + 1) * sizeof(Feature *));
        if (*Observations == NULL) {
            fprintf(stderr, "Error reallocing feature vectors\n");
            exit(1);
        }

        *Labels = (char **)realloc(*Labels, (last + 1) * sizeof(char *));
        if (*Labels == NULL) {
            fprintf(stderr, "Error reallocing label list\n");
            exit(1);
        }
    }

    // this will triger either adding an observation, or parsing out the header of the file.
    // depending on the value of last
    if (fp == NULL) {
        return (add_vector_noise(fp, Labels, Observations, Nobs, Nfeat, logging));
    } else {
        return (add_vector_file(fp, Labels, Observations, Nobs, Nfeat, logging));
    }
}

Node *create_nodes(Feature **observations, int nobs, int nfeat) {
    Node *o;
    int i;

    o = (Node *)calloc(nobs, sizeof(Node));
    if (o == NULL) {
        fprintf(stderr, "Error mallocing node\n");
        exit(1);
    }

    for (i = 0; i < nobs; ++i) {
        o[i].Centvec = (Feature *)malloc(nfeat * sizeof(Feature));
        if (o[i].Centvec == NULL) {
            fprintf(stderr, "Error mallocing node->Centvec\n");
            exit(1);
        }
#ifdef WSD
        o[i].Centvsq = (Feature *)malloc(nfeat * sizeof(Feature));
        if (o[i].Centvsq == NULL) {
            fprintf(stderr, "Error mallocing node->Centvsq\n");
            exit(1);
        }
#endif
    }

    return (o);
}

void clear_centroids(Node *o, int nobs, int nfeat) {
    int i, j;

    for (i = 0; i < nobs; ++i) {
        for (j = 0; j < nfeat; ++j) {
            o[i].Centvec[j] = 0.0;
#ifdef WSD
            o[i].Centvsq[j] = 0.0;
#endif
        }
        /* By definition the labels will change */
        for (j = 0; j < 128; ++j) {
            o[i].CharFreq[j] = 0;
        }
        o[i].nmemb = 0;
        o[i].nfreq = 0;
    }
}

void clear_node_classes(Node *o, int nobs) {
    int i, j;

    for (i = 0; i < nobs; ++i) {
        for (j = 0; j < 128; ++j) {
            o[i].CharFreq[j] = 0;
        }
        o[i].nfreq = 0;
    }
}

void clear_nodes(Node *o, Feature **observations, char **labels, int nobs, int nfeat) {
    int i, j;

    for (i = 0; i < nobs; ++i) {
        o[i].Childlist = NULL;
        o[i].Childlast = NULL;
        o[i].Sibnext = NULL;
        o[i].Sibback = NULL;
        o[i].parentdist = 0.0;
        o[i].Nchildren = 0;
        o[i].Pattvec = observations[i];
        o[i].Nfeat = nfeat;
        for (j = 0; j < nfeat; ++j) {
            o[i].Centvec[j] = 0.0;
#ifdef WSD
            o[i].Centvsq[j] = 0.0;
#endif
        }
        for (j = 0; j < 128; ++j) {
            o[i].CharFreq[j] = 0;
        }
        o[i].nmemb = 0;
        o[i].nfreq = 0;
        o[i].instance_id = i;
        o[i].had = 0;
        o[i].lev = -1;
        o[i].myparent = NULL;
        o[i].iclass = (int)labels[i][0];
        o[i].majority = 0;
    }
}

void shuffle(int *idx, int n) {
    int i, j, swp;
    
    for (i = 0; i < n; ++i) {
        idx[i] = i;
    }

    for (i = 0; i < n; i++) {
        j = (int)(((double)i + 1) * drand48());
        swp = idx[i];
        idx[i] = idx[j];
        idx[j] = swp;
    }

#ifdef DEBUGGING
    for (i = 0; i < n; i++) {
        fprintf(stdout, "%d ", idx[i]);
    }
    fprintf(stdout, "\n");
#endif
}

#define GRAPHICS
#ifdef GRAPHICS

#define XOFF 500.
#define YOFF 500.
#define XSCALE 60.
#define YSCALE 60.

#define ARROW_W 100.
#define ARROW_H 220.
#define ARROW_SCALE 6.

FILE *init_graph(char *filename) {
    FILE *fp;

    fp = fopen(filename, "w");

    fprintf(fp, "#FIG 3.2\n");
    fprintf(fp, "Landscape\n");
    fprintf(fp, "Center\n");
    fprintf(fp, "Inches\n");
    fprintf(fp, "Letter  \n");
    fprintf(fp, "100.00\n");
    fprintf(fp, "Single\n");
    fprintf(fp, "-2\n");
    fprintf(fp, "1200 2\n");

    return (fp);
}

void draw_arrow(FILE *fp, double x1, double y1, double x2, double y2, int icol, int iarr) {
    int ix1, iy1, ix2, iy2;
    double w, h, r, dx, dy;

    ix1 = (int)(XOFF + XSCALE * x1);
    iy1 = (int)(YOFF + YSCALE * y1);
    ix2 = (int)(XOFF + XSCALE * x2);
    iy2 = (int)(YOFF + YSCALE * y2);

    dx = ix2 - ix1;
    dy = iy2 - iy1;
    r = sqrt(dx * dx + dy * dy);

    w = sqrt(ARROW_W * r / ARROW_SCALE);
    h = sqrt(ARROW_H * r / ARROW_SCALE);

    /* icol 0 vs 4   iarr 3 or 1 */
    fprintf(fp, "2 1 0 2 %d 7 0 0 -1 0.000 0 0 -1 1 0 2\n", iarr);
    fprintf(fp, "	%d 1 1.00 %.2f %.2f\n", icol, w, h);
    fprintf(fp, "	 %d %d %d %d\n", ix1, iy1, ix2, iy2);
}

void draw_arrow_red(FILE *fp, double x1, double y1, double x2, double y2) { draw_arrow(fp, x1, y1, x2, y2, 4, 1); }

void draw_arrow_black(FILE *fp, double x1, double y1, double x2, double y2) { draw_arrow(fp, x1, y1, x2, y2, 3, 0); }

void close_graph(FILE *fp) {
    if (fp != NULL) {
        fclose(fp);
    }
}

void draw_node(Node *p, int irad, int ifill) {

    int ix1, iy1, i1, i2;

    ix1 = (int)(XOFF + XSCALE * p->Pattvec[0]);
    iy1 = (int)(YOFF + YSCALE * p->Pattvec[1]);

    if (p != NULL) {
        if (ifill == 0) {
            i1 = 7;
            i2 = -1;
        } else {
            i1 = 0;
            i2 = 20;
        }
        fprintf(fpg, "1 3 0 1 0 %d 0 0 %d 0.000 1 0.0000 %d %d %d %d %d %d 1275 1350\n", i1, i2, ix1, iy1, irad, irad,
                ix1, iy1);
    }
}

void draw_sibbling(Node *p) {
    double x1, y1, x2, y2;

    if (p != NULL) {
        if (p->Sibnext != NULL) {
            x1 = p->Pattvec[0];
            y1 = p->Pattvec[1];
            x2 = p->Sibnext->Pattvec[0];
            y2 = p->Sibnext->Pattvec[1];
            draw_arrow_red(fpg, x1, y1, x2, y2);
        }
    }
}

void draw_child(Node *p) {
    double x1, y1, x2, y2;

    if (p != NULL) {
        if (p->Childlist != NULL) {
            x1 = p->Pattvec[0];
            y1 = p->Pattvec[1];
            x2 = p->Childlist->Pattvec[0];
            y2 = p->Childlist->Pattvec[1];
            draw_arrow_black(fpg, x1, y1, x2, y2);
        }
    }
}
#endif

void list_siblings(Node *node, int lev, int *ntraversed, int logging) {
    Node *p = NULL;
    int nsib;

    if (node == NULL)
        return;

    p = node->Sibnext;

    if (p != NULL) {
        if (logging >= 3)
            fprintf(stdout, "sib: ");
        nsib = 1;
    } else {
        nsib = 0;
    }

    while (p != NULL) {
        if (logging >= 3) {
            fprintf(stdout, "%04d, ", p->instance_id);
        }
        p = p->Sibnext;
        ++nsib;
    }
    if (nsib != 0) {
        if (logging >= 3)
            fprintf(stdout, "\n");
    }

    if (logging >= 5)
        draw_sibbling(node);
    p = node->Sibnext;
    while (p != NULL) {
        if (!p->had)
            traverse_children(p, lev + 1, ntraversed, logging);
        if (logging >= 5)
            draw_sibbling(p);
        p = p->Sibnext;
    }
}

void traverse_children(Node *node, int lev, int *ntraversed, int logging) {
    Node *p;

    if (node == NULL)
        return;

    ++(*ntraversed);

    if (logging >= 5)
        draw_node(node, 30, 0);

    if (logging != 0) {
        fprintf(stdout, "Node id=%04d=%f lev=%d pd=%f parent=%04d nch=%d nsib=%d\n", node->instance_id,
                node->Pattvec[0], lev, node->parentdist, idNode(node->myparent), get_nchildren(node, 0),
                get_nsibnext(node, 0));
    }

    node->had = 1;
    node->lev = lev;

    list_siblings(node, lev, ntraversed, logging);

    if (logging >= 5)
        draw_child(node);
    p = node->Childlist;
    while (p != NULL) {
        if (!p->had)
            traverse_children(p, lev + 1, ntraversed, logging);
        if (logging >= 5)
            draw_child(p);
        p = p->Childlist;
    }
}

char most_likely_label(Node *o) {
    int j, mx, ibest;

    mx = -1;
    ibest = 32;
    dprintf("%04d lev %d nmem %d nfreq %d ", o->instance_id, o->lev, o->nmemb, o->nfreq);
    for (j = 0; j < 128; ++j) {
        if (o->CharFreq[j] > 0) {
            dprintf("[%c]=%d ", (char)j, o->CharFreq[j]);
        }

        if (o->CharFreq[j] > mx) {
            mx = o->CharFreq[j];
            ibest = j;
        }
    }
    if (mx == 0)
        ibest = o->iclass;
    dprintf("[%c]=%d!\n", (char)ibest, mx);
    return ((char)ibest);
}

void typer_centroids(Node *o, int Nnodes) {
    int i;
    for (i = 0; i < Nnodes; ++i) {
        o[i].majority = (int)most_likely_label(&o[i]);
    }
}

void clear_neighbours(int *iout, double *dout, int k, int *k_found) {
    int i;
    for (i = 0; i < k; ++i) {
        dout[i] = 9999999.;
        iout[i] = -1;
    }
    *k_found = 0;
}

void add_to_neighbourlist(double dnew, int idnode, int *iout, double *dout, int k, int kmin, int kmax, int *k_found) {
    int i, j;

    i = 0;
    while (dout[i] < dnew && i < *k_found && i < k) {
        ++i;
    }
    if (i < k) { /* point a i is worse */
        if (k > 1) {
            for (j = k - 1; j > i; --j) {
                dout[j] = dout[j - 1];
                iout[j] = iout[j - 1];
            }
        }
        dout[i] = dnew;
        iout[i] = idnode;
    }
    ++(*k_found);
    if (*k_found > k) {
        k = *k_found;
    }
}

void list_siblings_search(Node *node, int lev, int logging, double radius, double *vec, char *lab, int *iout,
                          double *dout, int *k_found, int mode) {
    Node *p = NULL, *pbest[1] = {NULL};
    double d, dmin, dbest_at_start;

    if (node == NULL)
        return;

    p = node->Sibnext;

    dmin = 999999999.;

    dbest_at_start = dout[0];

    while (p != NULL) {
#ifdef DIST_BOOKKEEPING
        if (!p->dcalced) {
            d = dist_match(vec, p, radius, MatchMethod);
            p->probedist = d;
            p->dcalced = 1;
        } else {
            d = p->probedist;
        }
#else
        d = dist_match(vec, p, radius, MatchMethod);
#endif
        if (d < dmin) {
            dmin = d;
            pbest[0] = p;
        }
        add_to_neighbourlist(d, idNode(p), iout, dout, K_KNN, Kmin_KNN, Kmax_KNN, k_found);

        p = p->Sibnext;
    }

    if (dmin <= dbest_at_start) {
        if (!pbest[0]->had) {
            traverse_search(pbest[0], lev + 1, logging, radius, vec, lab, iout, dout, k_found, mode);
        }
    }
}

void traverse_search(Node *node, int lev, int logging, double radius, double *vec, char *lab, int *iout, double *dout,
                     int *k_found, int mode) {
    Node *p;

    if (node == NULL)
        return;

    if (mode == TYPER_MODE) {
        ++(node->CharFreq[(int)lab[0]]);
        ++(node->nfreq);
    }

    if (logging != 0) {
        fprintf(stdout, "Node id=%04d=%f lev=%d pd=%f parent=%04d nch=%d nsib=%d\n", node->instance_id,
                node->Pattvec[0], lev, node->parentdist, idNode(node->myparent), get_nchildren(node, 0),
                get_nsibnext(node, 0));
    }

    node->had = 1;

    list_siblings_search(node, lev, logging, radius, vec, lab, iout, dout, k_found, mode);

    p = node->Childlist;
    while (p != NULL) {
        if (!p->had) {
            traverse_search(p, lev + 1, logging, radius, vec, lab, iout, dout, k_found, mode);
        }
        p = p->Childlist;
    }
}

void addto_centroid(Node *p, Node *node) {
    int i;

    for (i = 0; i < p->Nfeat; ++i) {
        node->Centvec[i] += p->Pattvec[i];
#ifdef WSD
        node->Centvsq[i] += p->Pattvec[i] * p->Pattvec[i];
#endif
    }

// #define DOUBLE_TYPING
#ifdef DOUBLE_TYPING
    ++(node->CharFreq[p->iclass]);
    ++(node->nfreq);
#endif

    ++(node->nmemb);
}

void list_siblings_making_centroids(Node *mother, Node *node, int lev, int logging) {
    Node *p;

    if (node == NULL)
        return;

    p = node->Sibnext;

    while (p != NULL) {
        if (!p->had) {
            traverse_making_centroids(mother, p, lev + 1, logging);
        }
        p = p->Sibnext;
    }
}

void traverse_making_centroids(Node *mother, Node *node, int lev, int logging) {
    Node *p;

    if (node == NULL)
        return;

    addto_centroid(node, mother);

    if (logging != 0) {
        fprintf(stdout, "Node id=%04d=%f lev=%d pd=%f parent=%04d nch=%d nsib=%d\n", node->instance_id,
                node->Pattvec[0], lev, node->parentdist, idNode(node->myparent), get_nchildren(node, 0),
                get_nsibnext(node, 0));
    }

    node->had = 1;
    node->lev = lev;

    list_siblings_making_centroids(mother, node, lev, logging);

    p = node->Childlist;
    while (p != NULL) {
        if (!p->had) {
            traverse_making_centroids(mother, p, lev + 1, logging);
        }
        p = p->Childlist;
    }
}

double sd(double avgsq, double avg) {
    double r;

    r = avgsq - avg * avg;
    if (r < 0.00001)
        r = 0.00001;
    return (sqrt(r));
}

void average_centroids(Node *o, int Nnodes, int nfeat) {
    int i, j, n;

    for (i = 0; i < Nnodes; ++i) {
        n = o[i].nmemb;
        if (n >= 1) {
            dprintf("Node %d=%04d has %d nmemb (ok)\n", o[i].instance_id, i, n);
            for (j = 0; j < nfeat; ++j) {
                o[i].Centvec[j] /= (double)n;
#ifdef WSD
                o[i].Centvsq[j] /= (double)n;
                o[i].Centvsq[j] = sd(o[i].Centvsq[j], o[i].Centvec[j]);
#endif
            }
        } else {
            printf("Node %d has %d nmemb!!! (err) \n", i, n);
        }
    }
}

void clear_traversal(Node *array, int Nnodes) {
    int i;

    for (i = 0; i < Nnodes; ++i) {
        array[i].had = 0;
#ifdef DIST_BOOKKEEPING
        array[i].dcalced = 0;
        array[i].probedist = 999999.;
#endif
    }
}

Node *find_highest_parent(Node *p) {
    Node *t;

    t = p;
    while (1) {
        if (t->myparent == NULL)
            break;
        t = t->myparent;
    }
    return (t);
}

void find_child_ref(Node *anode, Node *array, int Nnodes) {
    int i;
    Node *p;

    printf("     %04d is mentioned as child in:\n", idNode(anode));
    for (i = 0; i < Nnodes; ++i) {
        p = array[i].Childlist;
        while (p != NULL) {
            if (p == anode) {
                printf("%04d ", idNode(&array[i]));
            }
            p = p->Childlist;
        }
    }
    printf("\n");
}

void find_sibling_ref(Node *anode, Node *array, int Nnodes) {
    int i;
    Node *p;

    printf("     %04d is mentioned as sibling in:\n", idNode(anode));
    for (i = 0; i < Nnodes; ++i) {
        p = array[i].Sibnext;
        while (p != NULL) {
            if (p == anode) {
                printf("%04d ", idNode(&array[i]));
            }
            p = p->Sibnext;
        }
    }
    printf("\n");
}

int check_traversal(Node *array, int Nnodes) {
    int i, nchildren, pid, pid_top, ok;
    Node *parent, *parent_top;

    nchildren = 0;

    ok = 1;

    for (i = 0; i < Nnodes; ++i) {
        if (array[i].had)
            ++nchildren;
    }
    printf("Nnodes=%d traversed Nchildren=%d\n", Nnodes, nchildren);

    if (Nnodes != nchildren) {
        printf("Incomplete linking!\n");
        for (i = 0; i < Nnodes; ++i) {
            if (!array[i].had) {
                parent = array[i].myparent;
                pid = idNode(parent);
                parent_top = find_highest_parent(parent);
                pid_top = idNode(parent_top);

                printf("Node id=%04d parent=%04d not visited. top parent=%04d\n", array[i].instance_id, pid, pid_top);
                find_child_ref(&array[i], array, Nnodes);
                find_sibling_ref(&array[i], array, Nnodes);

                ok = 0;
            }
        }
    }
    return (ok);
}

int ntraversed(Node *array, int Nnodes) {
    int i, nchildren;

    nchildren = 0;

    for (i = 0; i < Nnodes; ++i) {
        if (array[i].had)
            ++nchildren;
    }
    return (nchildren);
}

double offspring_density(Node *array, int Nnodes) {
    int i, nfilled = 0;

    for (i = 0; i < Nnodes; ++i) {
        if (array[i].Nchildren != 0) {
            ++nfilled;
        }
    }
    return ((double)(nfilled + 1) / (double)Nnodes);
}

double sibling_density(Node *array, int Nnodes) {
    int i, nfilled = 0;

    for (i = 0; i < Nnodes; ++i) {
        if (array[i].Sibnext != NULL) {
            ++nfilled;
        }
    }
    return ((double)nfilled / (double)Nnodes);
}

void usage() {
    fprintf(stderr, "Usage: boom [R] [seed] feat.dat|-[int Nrandom samples] [logging "
                    "0-5 ] [- or test.dat] [method] k kmin kmax nmin\n");
    fprintf(stderr, "       [method] 1=1NN_ABS 2=1NN_REL 4=NCENT\n");
    fprintf(stderr, "Optional additional argument [num_classes, 0.0 < fraction <= 1.0]\n");
    fflush(stderr);
    exit(1);
}

Feature **init_vectors() {
    Feature **v;

    v = (Feature **)malloc(sizeof(Feature *));
    if (v == NULL) {
        fprintf(stderr, "Error mallocing feature vectors\n");
        exit(1);
    }
    return (v);
}

char **init_labels() {
    char **v;

    v = (char **)malloc(sizeof(char *));
    if (v == NULL) {
        fprintf(stderr, "Error mallocing labels\n");
        exit(1);
    }
    return (v);
}

FILE *fopen_infile(char *filename, char ***Labels, Feature ***Observations) {
    FILE *fp;

    if (filename[0] == '-') {
        fp = NULL;
        NSAMP_IF_RANDOM = -atoi(filename);

    } else {
        fp = fopen(filename, "r");
        if (fp == NULL) {
            fprintf(stderr, "Error opening inputfile [%s]\n", filename);
            exit(1);
        }
    }

    *Observations = init_vectors();
    *Labels = init_labels();
    return (fp);
}

char vote_knn(char **Labels, int *irec, int k_knn, int *nvotes) {
    int i, freqs[128];
    int ichr, maxfreq, ibest;

    for (i = 0; i < 128; ++i) {
        freqs[i] = 0;
    }
    for (i = 0; i < k_knn; ++i) {
        ichr = (int)Labels[irec[i]][0];
        ++freqs[ichr];
    }
    maxfreq = 0;
    ibest = 0;
    for (i = 0; i < 128; ++i) {
        if (freqs[i] > maxfreq) {
            maxfreq = freqs[i];
            ibest = i;
        }
    }
    //   fprintf(stdout,"k=%dNN best = %c f=%d\n", k_knn, (char) ibest,
    //   maxfreq);
    *nvotes = maxfreq;
    return ((char)ibest);
}

char votor(char a, char b, char c) {
    int i, maxfreq, ibest;
    char *lst;

    lst = (char*)calloc(128, sizeof(char));

    ++lst[(int)a];
    ++lst[(int)b];
    ++lst[(int)c];

    ibest = 0;
    maxfreq = 0;
    for (i = 32; i <= (int)'z'; ++i) {
        if (lst[i] > maxfreq) {
            ibest = i;
            maxfreq = lst[i];
        }
    }

    free(lst);
    return ((char)ibest);
}

char getRecognizedLabel(Node *o, char **Labels, int *irec, int k_knn) {
    char label;

    if (k_knn == 1) {
        if (MatchMethod <= MATCH_NCENT_1NN)
            label = Labels[irec[0]][0];
        else
            label = (char)o[irec[0]].majority;
    } else {
        if (MatchMethod <= MATCH_NCENT_1NN) {
            int n_knn_votes = 0;
            label = vote_knn(Labels, irec, k_knn, &n_knn_votes);
        } else
            label = (char)o[irec[0]].majority;
    }

    return label;
}

int okLabel(Node *o, int Nnodes, char **tLabels, int itest, char **Labels, int *irec, int k_knn) {
    int iret, n_knn_votes = 0, n_memb_votes = 0;
    char test_label, recog_label;
    double p_knn_votes = 0.0, p_memb_votes = 0.0;

    test_label = tLabels[itest][0];

    if (k_knn == 1) {
        if (MatchMethod <= MATCH_NCENT_1NN) {
            recog_label = Labels[irec[0]][0];

        } else {
            recog_label = (char)o[irec[0]].majority;
        }

    } else {
        if (MatchMethod <= MATCH_NCENT_1NN) {
            recog_label = vote_knn(Labels, irec, k_knn, &n_knn_votes);
        } else {
            recog_label = (char)o[irec[0]].majority;
        }
    }

    if (test_label == recog_label) {
        iret = 1;
    } else {
        iret = 0;
    }
    return (iret);
}

#ifdef Kullback

void compute_Ppix(Feature **Observations, int Nobs, int nfeat, Feature *Pa) {
    int i, j;
    for (j = 0; j < nfeat; ++j) {
        Pa[j] = 0.0;
    }
    for (i = 0; i < Nobs; ++i) {
        for (j = 0; j < nfeat; ++j) {
            Pa[j] += Observations[i][j];
        }
    }
    for (j = 0; j < nfeat; ++j) {
        Pa[j] /= (double)Nobs;
    }
}

void compute_Pclass(char **Labels, int Nobs, int nclass, Feature *Pb, int minclass, int maxclass) {
    int i, j;
    for (j = minclass; j <= maxclass; ++j) {
        Pb[j] = 0.0;
        for (i = 0; i < Nobs; ++i) {
            if ((int)Labels[i][0] == j) {
                Pb[j] += 1.;
            }
        }
        Pb[j] /= Nobs;
    }
}

void Kullback_relative_information(Feature **Observations, char **Labels, int Nobs, Feature *wkb, int nfeat, char from,
                                   char to) {
    int i, j, k, kk, minclass, maxclass;
    Feature *Pa;
    Feature *Pb;
    Feature Pab, Kull, pp, PaPb;

    minclass = (int)from;
    maxclass = (int)to;

    Pa = (Feature *)calloc(nfeat, sizeof(Feature));
    Pb = (Feature *)calloc(128, sizeof(Feature));

    compute_Ppix(Observations, Nobs, nfeat, Pa);
    compute_Pclass(Labels, Nobs, 128, Pb, minclass, maxclass);

    for (j = 0; j < nfeat; ++j) {
        Kull = 0.0;
        for (k = minclass; k <= maxclass; ++k) {
            Pab = 0.0;
            for (i = 0; i < Nobs; ++i) {
                kk = (int)Labels[i][0];
                if (k == kk) {
                    Pab += Observations[i][j];
                }
            }
            Pab = Pab / Nobs;
            PaPb = (Pa[k] * Pb[j]);
#define Plow 0.00000001
            if (PaPb <= Plow)
                PaPb = Plow;

            pp = Pab / PaPb;
            if (pp <= Plow)
                pp = Plow;
            Kull += Pab * log(pp);
        }
        wkb[j] = Kull;
        if (wkb[j] < 0.0) {
            Wkb[j] = 0.0;
        } else {
            Wkb[j] = 1.0;
        }
        printf("Feature j=%d Kull=%.2f %.4f\n", j, Kull, Wkb[j]);
    }
    free(Pa);
    free(Pb);
}

#endif

int main(int argc, char *argv[]) {
    char filename[1000];
    const char* filename_fmt = "output/boom-r=%f-iter=%d.%s";

    static Feature **Observations = NULL;
    static char **Labels = NULL;
    int Nobs, Nfeat;

    static Feature **tObservations;
    static char **tLabels;
    int Nobs_t, Nfeat_t;

    int uses_fraction = 0;
    int num_classes = 10;
    double dataset_fraction = 1.0;

    int file_iteration = 0;
    int j, k, logging, ok, iter, ncorrect, ntrav, k_found, istat = 0, visited;
    long int seed;
    Node *array = NULL, *first = NULL;
    int *idx, iout[MAXKNN];
    double radius, dout[MAXKNN];
    double tpu_create, tpu_trav, tpu_class;
    FILE *fp;
#ifdef Kullback
    double *wkb;
#endif

    setbuf(stdout, NULL);
    if (argc != 11 && argc != 12) {
        usage();
    }

    radius = atof(argv[1]);
    seed = (long int)atoi(argv[2]);
    logging = atoi(argv[4]);
    MatchMethod = atoi(argv[6]);
    if (MatchMethod < 1 || MatchMethod > 5) {
        usage();
    }
    K_KNN = atoi(argv[7]);
    Kmin_KNN = atoi(argv[8]);
    Kmax_KNN = atoi(argv[9]);
    Nmin_NCENT = atoi(argv[10]);

    if (logging >= 5) {
        /* xfig format output file */
        fpg = init_graph("boom.fig");
    }
    
    if (argc == 12) {
        file_iteration = atoi(argv[11]);
    }
    

    srand48(seed);

    fp = fopen_infile(argv[3], &Labels, &Observations);

    // for dataset fraction we do first load all the observations
    Nobs = 0;
    while (add_vector(fp, &Labels, &Observations,  Nobs, &Nfeat, logging)) {
        ++Nobs;
    }
    fclose(fp);

    // Alloc the index list and write numbers from 0-Nobs to it
    idx = (int *)malloc(Nobs * sizeof(int));

#ifdef Kullback
    // This has NOT been adapted to use class labels & fractions
    wkb = (Feature *)calloc(Nfeat, sizeof(Feature));
    Kullback_relative_information(Observations, Labels, Nobs, wkb, Nfeat, '0', '9');
#endif

    array = create_nodes(Observations, Nobs, Nfeat);

    ok = 0;
    iter = 0;
    visited = 0;
    printf("Start\n");
    while (!ok && iter < MAXITER) {
        visited = 0;

        tpu_create = timing(TIMING_START, 0, "", 0);

        shuffle(idx, Nobs);

        clear_nodes(array, Observations, Labels, Nobs, Nfeat);

        first = &array[idx[0]];

        printf("\nIter %d\n", iter);
        for (k = 0; k < Nobs; ++k) {
            if (logging > 0) {
                printf("Iter %d samp %d\n", iter, k);
            }



            istat = insert_pattern_portegys(&array[idx[k]], first, radius, Nobs, &visited);
            printf("istat=%d main_insert_pattern_portegys();\n", istat);
            if (istat == PROBLEM) {
                break;
            }
        }

        if (istat != PROBLEM) {

            tpu_create = timing(TIMING_STOP, Nobs, "created tree", 0);

            if (logging >= 2) {
                for (k = 0; k < Nobs; ++k) {
                    print_node(&array[idx[k]], k);
                }
            }

            clear_traversal(array, Nobs);

            if (logging >= 5)
                draw_node(first, 80, 1);

            tpu_trav = timing(TIMING_START, 0, "", 1);

            ntrav = 0;
            traverse_children(first, 0, &ntrav, logging);
            printf("ntrav=%d\n", ntrav);

            tpu_trav = timing(TIMING_STOP, Nobs, "traversed tree", 1);

            ok = check_traversal(array, Nobs);
            ++iter;

            if (MatchMethod >= MATCH_NCENT_1NN) {
                printf("Creating centroids, root =%04d\n", idNode(first));
                clear_centroids(array, Nobs, Nfeat);
                for (k = 0; k < Nobs; ++k) {
                    clear_traversal(array, Nobs);
                    traverse_making_centroids(&array[k], &array[k], 0, 0);
                }
                average_centroids(array, Nobs, Nfeat);

                printf("Creating node types, root =%04d\n", idNode(first));

#ifndef DOUBLE_TYPING
                clear_node_classes(array, Nobs);
#endif
                for (k = 0; k < Nobs; ++k) {
                    clear_neighbours(iout, dout, K_KNN, &k_found);
                    clear_traversal(array, Nobs);

                    traverse_search(first, 0, logging, radius, Observations[k], Labels[k], iout, dout, &k_found,
                                    TYPER_MODE);
                }
                typer_centroids(array, Nobs);
            }

            if (!does_folder_exist("output")) {
                create_folder("output");
            }

            

            sprintf(filename, filename_fmt, file_iteration, radius, "out");
            save_tree(filename, array, Nobs, first);
            sprintf(filename, filename_fmt, file_iteration, radius, "hist");
            save_hist(filename, array, Nobs, first);

            if (logging >= 5)
                close_graph(fpg);

            printf("R= %f offspring density= %f sibling density= %f fanout= %f\n", radius,
                   offspring_density(array, Nobs), sibling_density(array, Nobs), fanout_factor(first, Nobs));
        } else {
            printf("Retry, Visited=%d\n", visited);
        }
    }
    
    printf("Tree created, points Visited=%d\n", visited);
    
    // Saves a file which maps the class to each node index
    sprintf(filename, filename_fmt, file_iteration, radius, "map");
    save_node_class_mapping(filename, array, Nobs);

    if (argv[5][0] != '-') {
        fp = fopen_infile(argv[5], &tLabels, &tObservations);

        Nobs_t = 0;
        while (add_vector(fp, &tLabels, &tObservations, Nobs_t, &Nfeat_t, logging)) {
            if (Nfeat_t != Nfeat) {
                fprintf(stderr, "test set Nfeat=%d training set Nfeat=%d\n", Nfeat_t, Nfeat);
                exit(1);
            }

            ++Nobs_t;
        }
        fclose(fp);

        tpu_class = timing(TIMING_START, 0, "", 2);
        Ndist = 0;
        ncorrect = 0;

        sprintf(filename, filename_fmt, file_iteration, radius, "det");

        // We open file here to write the detections to
        FILE *fpt = fopen(filename, "w");
        if (fpt == NULL) {
            fprintf(stderr, "Error opening output file for detections");
            exit(1);
        }
        fprintf(fpt, "Sample,SampleLabel,RecognizedLabel\n");

        for (k = 0; k < Nobs_t; ++k) {
            clear_neighbours(iout, dout, K_KNN, &k_found);
            clear_traversal(array, Nobs);

            traverse_search(first, 0, logging, radius, tObservations[k], "*", iout, dout, &k_found, TEST_MODE);

            if (logging > 0) {
                j = 0;
                printf("%s ntrav=%d\n", tLabels[k], ntraversed(array, Nobs));
                while (j < K_KNN && iout[j] >= 0) {
                    printf("%d %c? %s %.2f\n", j, tLabels[k][0], Labels[iout[j]], dout[j]);
                    ++j;
                }
            }
            
            // We also write the found mapping here, so we can use that to build a confusion matrix later on
            char *test_entry = tLabels[k];
            char test_label = tLabels[k][0];
            char recognized_label = getRecognizedLabel(array, Labels, iout, K_KNN);
            fprintf(fpt, "%s,%c,%c\n", test_entry, test_label, recognized_label);

            if (test_label == recognized_label) 
                ++ncorrect;
            // if (okLabel(array, Nobs, tLabels, k, Labels, iout, K_KNN))
            //     ++ncorrect;
        }

        tpu_class = timing(TIMING_STOP, Nobs_t, "recognition", 2);
        printf("R= %f Accu= %.2f %% Ndist=%d tpu %.0f. us\n", radius, 100. * (double)ncorrect / (double)Nobs_t, Ndist, tpu_class);
        fclose(fpt);
    }
    return (0);
}
