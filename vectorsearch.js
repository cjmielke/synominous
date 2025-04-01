import { PGlite } from "https://cdn.jsdelivr.net/npm/@electric-sql/pglite@0.2.11/dist/index.js";
import { vector } from "https://cdn.jsdelivr.net/npm/@electric-sql/pglite@0.2.11/dist/vector/index.js";

var tiles_path; var tiles_suffix;
tiles_path = 'data/tiles/0'; tiles_suffix = '.png';
tiles_path = 'data/greyscale_tiles'; tiles_suffix = '.png';
tiles_path = 'https://cjmielke.github.io/synominous_grey_tiles/tiles/'; tiles_suffix = '-fs8.png';


const LOCAL_DB_NAME = `vector-search-demo`; // local persistent db

const status = $('#status');
const progressbar = $('#progressbar');


async function fetchAndProcessGzipJson(url) {
  try {
    const response = await fetch(url, {
      headers: {
        'Accept-Encoding': 'gzip' // Explicitly request gzipped content
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Check if the server sent a Content-Encoding header indicating gzip
    if (response.headers.get('Content-Encoding') === 'gzip') {
      // Use DecompressionStream to handle the gzipped response
      const ds = new DecompressionStream('gzip');
      const decompressedStream = response.body.pipeThrough(ds);
      const decompressedText = await new Response(decompressedStream).text();
      const jsonData = JSON.parse(decompressedText);
      return jsonData;
    } else {
      // If not gzipped, assume it's plain text JSON
      const jsonData = await response.json();
      return jsonData;
    }
  } catch (error) {
    console.error("Error fetching or processing data:", error);
    throw error; // Re-throw to allow handling by the caller
  }
}


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

//CREATE INDEX IF NOT EXISTS embeddings_embedding_idx ON embeddings USING hnsw (embedding vector_ip_ops);

async function initSchema(db, embsize) {
    await db.exec(`
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS embeddings (
          synapse INT NOT NULL UNIQUE,
          neurotransmitter INT,
          embedding VECTOR(${embsize})
        );
        
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
    status.html(`“Building embedding database (only on first load) : ☑ Downloaded ☑ Decompressed | Inserting ...`);
    const chunks = chunkArray(embeddings, 100);
    chunks.forEach(function (embeddings){
    })
    for (const [i, embeddings] of chunks.entries()) {
        progressbar.css({'width': `${i}%`});
        const pg_entries = embeddings.map((entry) => {
            //const neurotransmitter = entry.text.replaceAll(`'`, ``);
            return `\t('${entry.s}', '${entry.n}', '${JSON.stringify(entry.v)}')`;
        }).join(",\n");
        //console.log(pg_entries);
        await db.exec(`insert into embeddings (synapse, neurotransmitter, embedding) values ${pg_entries};`);
    }
    progressbar.css({'width': `100%`});
    status.html('Counting rows');
    console.dir(await db.query(`SELECT COUNT(*) FROM embeddings;`), {depth: null,});
    const result = await db.query(`SELECT COUNT(*) FROM embeddings;`);
    console.log('Done loading database - rowcount :', JSON.stringify(result))
}

async function buildDatabase(){

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
//metric = '<#>'; order = 'DESC';
//metric = '<->'; order = 'ASC';
//metric='<=>'; order='ASC';


async function search({db, embedding, match_threshold = 50.0, limit = 10,}) {
    var query = `
        SELECT * FROM (
            SELECT 
                synapse, 
                neurotransmitter, 
                embedding ${metric} $1 AS score,
                rank() over (partition by neurotransmitter order by embeddings.embedding ${metric} $1 ${order}) as rank
            FROM embeddings
        ) t
        WHERE rank < 5
    `;
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
        var color='gray';
        if(row.synapse==synapse_id){color='blue'};
        const thumb = $(`
            <div class="mb-2">
                <div class="bg-${color}-100 p-4 rounded-lg shadow-md max-h-50 overflow-y-auto break-words whitspace-pre-wrap">
                    <p class="text-xs font-semibold text-gray-700">(score: ${row.score.toFixed(4)})</p>
                    <!--<p class="text-sm text-gray-600 mt-1 max-h-10 overflow-auto">${row.neurotransmitter}</p>-->
                    <img alt="${row.synapse}" src="${tiles_path}/${row.synapse}${tiles_suffix}" />
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

    // L2 (<->), L1 (<+>), Inner product (<#>), cosine distance (<=>)
    const metric_select = $("#metric")
    metric_select.on("change", async function () {
        console.log(this.value);
        switch (this.value){
            case 'L2':
                metric='<->'; order='ASC'; break;
            case 'L1':
                metric='<+>'; order='ASC'; break;
            case 'COS':
                metric='<=>'; order='ASC'; break;
            case 'IP':
                metric='<#>'; order='ASC'; break
        }
    })

    // set default distance metric
    metric_select.val('COS').change();
    console.log(`metric : ${metric}   order: ${order}`)

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
        status.html(`“Building embedding database (only on first load) : ☐ Downloaded ☐ Decompressed | Inserting ...`);
        const response = await fetch("embeddings.json.gz");
        if (!response.ok) {throw new Error(`Response status: ${response.status}`);}
        // handle gz-compressed version
        status.html(`“Building embedding database (only on first load) : ☑ Downloaded ☐ Decompressed | Inserting ...`);
        const ds = new DecompressionStream('gzip');
        const decompressedStream = response.body.pipeThrough(ds);
        const decompressedText = await new Response(decompressedStream).text();

        //const J = await response.json();
        const J = JSON.parse(decompressedText);
        status.html(`“Building embedding database (only on first load) : ☑ Downloaded ☑ Decompressed | Inserting ...`);
        const embeddings = J['embeddings'];
        //console.log(embeddings);
        first_synapse = embeddings[0].s;
        console.log('Done loading database', first_synapse)
        const embsize = embeddings[0].v.length
        await initSchema(db, embsize);
        await insertEmbeddings({ embeddings: embeddings, db });
        //await $.get("embeddings.json").then(function(d){
        //}).catch(function (err){
        //    console.log(err);
        //    alert(err.message);
        //})

    }//else{
    first_synapse = await getFirstRow(db,'embeddings');
    first_synapse = Number(first_synapse.synapse);
    //$('#progressbar').css({'width': '100%'});
    //$('#status').html('Done!');
    $('.progressbar').hide();
    $('#status').hide();
    show_similar_tiles(first_synapse, db);
    //}



});
