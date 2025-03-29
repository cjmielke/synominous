import { PGlite } from "https://cdn.jsdelivr.net/npm/@electric-sql/pglite@0.2.11/dist/index.js";
import { vector } from "https://cdn.jsdelivr.net/npm/@electric-sql/pglite@0.2.11/dist/vector/index.js";

const LOCAL_DB_NAME = `vector-search-demo`; // local persistent db

function chunkArray(arr, n) {
    const size = Math.ceil(arr.length / n);
    return Array.from({ length: n }, (v, i) =>
        arr.slice(i * size, i * size + size)
    );
}

// pglite/pgvector integration courtesy of https://supabase.com/blog/in-browser-semantic-search-pglite
let dbInstance = null;
async function getDB() {
    if (dbInstance) {return dbInstance;}
    const db = new PGlite(`idb://${LOCAL_DB_NAME}`, {
        extensions: {vector,},
    });
    await db.waitReady;
    dbInstance = db;
    return db;
}



async function initSchema(db, embsize) {
    await db.exec(`
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS embeddings (
          synapse TEXT NOT NULL UNIQUE,
          neurotransmitter TEXT,
          embedding VECTOR(${embsize})
        );
        CREATE INDEX IF NOT EXISTS embeddings_embedding_idx ON embeddings USING hnsw (embedding vector_ip_ops);
    `);
}

async function countRows(db, table) {
    const res = await db.query(`SELECT COUNT(*) FROM ${table};`);
    return res.rows[0].count;
}

async function getFirstRow(db, table) {
    const res = await db.query(`SELECT * FROM ${table} LIMIT 1;`);
    return res.rows[0];
}

async function insertEmbeddings({ embeddings, db }) {
    const chunks = chunkArray(embeddings, 100);
    chunks.forEach(function (embeddings){
    })
    for (const [i, embeddings] of chunks.entries()) {
        $('#progressbar').css({'width': `${i}%`});
        const pg_entries = embeddings.map((entry) => {
            //const neurotransmitter = entry.text.replaceAll(`'`, ``);
            return `\t('${entry.s}', '${entry.n}', '${JSON.stringify(entry.v)}')`;
        }).join(",\n");
        //console.log(pg_entries);
        await db.exec(`insert into embeddings (synapse, neurotransmitter, embedding) values ${pg_entries};`);
    }
    console.dir(await db.query(`SELECT COUNT(*) FROM embeddings;`), {depth: null,});
    const result = await db.query(`SELECT COUNT(*) FROM embeddings;`);
    console.log('Done loading database - rowcount :', JSON.stringify(result))
}


async function getVector(db, eid) {
    try {
        const res = await db.query(
            `SELECT synapse, embedding FROM embeddings WHERE embeddings.synapse = $1;`,
            [Number(eid),],
        );
        console.log({ debug_search_res: { res } });
        return JSON.parse(res.rows[0].embedding);
    } catch (error) {
        console.error("Search Error:", error);
        //alert("Error during search operation.");
        return [];
    }
}


//SELECT synapse, neurotransmitter, embedding <#> $1 AS score FROM embeddings
//WHERE embeddings.embedding <#> $1 < $2
//ORDER BY embeddings.embedding <#> $1
//LIMIT $3;

//https://stackoverflow.com/questions/7613785/postgresql-top-n-entries-per-item-in-same-table
// L2 (<->), L1 (<+>), Inner product (<#>), cosine distance (<=>)

var metric, order;
metric = '<#>'; order = 'DESC';
metric = '<->'; order = 'ASC';

const query = `
    SELECT * FROM (
        SELECT 
            synapse, 
            neurotransmitter, 
            embedding ${metric} $1 AS score,
            rank() over (partition by neurotransmitter order by embeddings.embedding ${metric} $1 ${order}) as rank
        FROM embeddings
    ) t
    WHERE rank < 10
`;

async function search({db, embedding, match_threshold = 50.0, limit = 10,}) {
    try {
        console.log(query);
        const res = await db.query(query, [JSON.stringify(embedding),],);
        console.log({ debug_search_res: { embedding, res } });
        return res.rows;
    } catch (error) {
        console.error("Search Error:", error);
        alert("Error during search operation.");
        return [];
    }
}



const neurotransmitters = {
    0:"acetylcholine",
    1:"gaba",
    2:"dopamine",
    3:"glutamate",
    4:"serotonin",
    5:"octopamine"
};



async function show_similar_tiles(synapse_id, db){
    console.log('asking database for embedding ', synapse_id);
    const queryEmbedding = await getVector(db, synapse_id);
    console.log('got embedding : ', queryEmbedding);

    const searchResults = await search({ db, embedding: queryEmbedding });

    // empty the tile columns
    Object.values(neurotransmitters).forEach(n => {$('#'+n).empty();})
    // place similar tiles into proper column
    searchResults.forEach((row) => {
        const thumb = $(`
            <div class="mb-2">
                <div class="bg-gray-100 p-4 rounded-lg shadow-md max-h-50 overflow-y-auto break-words whitspace-pre-wrap">
                    <p class="text-xs font-semibold text-gray-700">(score: ${row.score.toFixed(4)})</p>
                    <!--<p class="text-sm text-gray-600 mt-1 max-h-10 overflow-auto">${row.neurotransmitter}</p>-->
                    <img alt="${row.synapse}" src="tiles/0/${row.synapse}.png" />
                </div>
            </div>`);

        thumb.on("click", function (ev){show_similar_tiles(row.synapse, db);})

        const neurotransmitter = '#'+neurotransmitters[Number(row.neurotransmitter)];
        //console.log(neurotransmitter, row);
        $(neurotransmitter).append(thumb);
    });
}

$(document).ready(async function () {

    const db = await getDB();

    $("#drop_table").on("click", async function () {
        await db.exec('DROP TABLE IF EXISTS embeddings;');
        alert('cleared');
    })

    var rowCount=0;
    try{
        rowCount = await countRows(db, "embeddings");
        console.log(rowCount, 'rows');
    }catch(err){
        console.log(err.message);
    }

    var first_synapse=0;
    if(rowCount<1000){
        console.log('fetching embeddings');
        await $.get("embeddings.json").then(function(d){
            first_synapse = d.embeddings[0].s;
            console.log('Done loading database', first_synapse)
            const embsize = d.embeddings[0].v.length
            initSchema(db, embsize);
            insertEmbeddings({ embeddings: d.embeddings, db });
            const rowCount = countRows(db, "embeddings");
            console.log(rowCount, 'rows after insert');
        }).catch(function (err){
            console.log(err);
            alert(err.message);
        })

    }else{
        first_synapse = await getFirstRow(db,'embeddings');
        first_synapse = Number(first_synapse.synapse);
    }

    show_similar_tiles(first_synapse, db)

});
